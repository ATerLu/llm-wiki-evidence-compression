"""Measure a human-checked upper-bound diagnostic for bypassing remote rerank."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean, median
from time import perf_counter

from baseline import sentence_blocks
from cloud_experiment import BudgetTokenizer, FULL_LIMIT
from cloud_reader import CloudReader, MODEL
from external_compression import RERANK_MODEL, dump, read_question
from paired import score_row
from prepare import adapt, jsonl, question_groups, read_jsonl
from semantic_bridge_compile import prior_ids, raw_rows
from semantic_view_pilot import BUDGET, assemble
from wiki_bridge import span_text

SEED = 20261122
N = 12
METHODS = ('raw_full', 'full_rerank', 'manual_short', 'warm_view_route')


def freeze(output: Path) -> None:
    rows = raw_rows('musique')
    families = {qid: f for f, ids in question_groups(rows, 'musique').items() for qid in ids}
    used = prior_ids('musique')
    reserved = json.loads(Path('runs/semantic-restoration-512/reuse-rescreen.json').read_text(encoding='utf-8'))
    used.update(q for pair in reserved['sample'] for q in pair['ids'])
    excluded = {families[q] for q in used if q in families}
    index = defaultdict(list)
    for row in rows:
        if families[row['id']] in excluded or len(row['question_decomposition']) != 2:
            continue
        first, second = row['question_decomposition']
        if re.search(r'#\d+', first['question']) or not re.search(r'#1\b', second['question']):
            continue
        paragraphs = {p['idx']: p for p in row['paragraphs']}
        shared = paragraphs[first['paragraph_support_idx']]
        index[(families[row['id']], str(first['id']), shared['title'], shared['paragraph_text'])].append(row)
    pairs = defaultdict(dict)
    for (family, step, title, text), group in index.items():
        for a, b in combinations(sorted(group, key=lambda r: r['id']), 2):
            if a['question'] != b['question'] and a['answer'].strip().casefold() != b['answer'].strip().casefold():
                pairs[family].setdefault((a['id'], b['id']), (step, title, text))
    rng, frozen = random.Random(SEED), []
    for family in rng.sample(sorted(pairs), N):
        ids = rng.choice(sorted(pairs[family]))
        step, title, text = pairs[family][ids]
        first, second = rng.sample(ids, 2)
        frozen.append({'family_id':family, 'first':first, 'second':second, 'step':step,
                       'shared_title':title, 'shared_text':text})
    output.mkdir(parents=True)
    dump(output / 'freeze.json', {'seed':SEED, 'excluded_families':sorted(excluded), 'candidate_families':len(pairs),
                                 'pairs':frozen})


def inspect(output: Path) -> None:
    pairs = json.loads((output / 'freeze.json').read_text(encoding='utf-8'))['pairs']
    raw = {r['id']: r for r in raw_rows('musique')}
    for pair in pairs:
        first, second = (raw[pair[k]] for k in ('first', 'second'))
        q, label, docs = adapt(second, 'musique', pair['family_id'])
        print('\n', q['id'], 'warm:', first['question'], '\nquery:', q['question'], 'gold:', label['answer'])
        print('steps:', [(s['question'],s['answer'],s['paragraph_support_idx']) for s in second['question_decomposition']])
        for i, doc in enumerate(docs):
            if q['candidates'][i]['key'] in label['support']:
                print(i, doc['title'])
                for s in doc['spans']:
                    j = s['index']
                    print(' ', j, span_text(doc, j).strip())


def pack(tokenizer, docs, spans):
    selected = defaultdict(set)
    for i, j in spans:
        selected[i].add(j)
    text = sentence_blocks(docs, selected)
    return {'selected_blocks':sorted((i, sorted(js)) for i, js in selected.items()),
            'evidence':text, 'evidence_tokens':len(tokenizer.tokenize(text))}


def request(query, docs, method, evidence):
    budget = FULL_LIMIT if method == 'raw_full' else BUDGET
    if evidence['evidence_tokens'] > budget:
        raise ValueError(f'{query["id"]} {method} exceeds {budget}')
    return {'dataset':'musique', 'query_id':query['id'], 'family_id':query['family_id'],
            'method':method, 'question':query['question'], 'candidate_ids':[d['id'] for d in docs],
            'budget':budget, **evidence}


def store_shared(path, docs, warm, second_docs, spans):
    warm_spans = {(docs[i]['version'], j) for i, js in warm['selected_blocks'] for j in js}
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE proof (version TEXT, span INTEGER, PRIMARY KEY(version,span))')
        for i, j in spans:
            version = second_docs[i]['version']
            if (version, j) in warm_spans:
                db.execute('INSERT INTO proof VALUES (?,?)', (version, j))
        db.commit()
        return all(db.execute('SELECT 1 FROM proof WHERE version=? AND span=?', (second_docs[i]['version'], j)).fetchone()
                   for i, j in spans)


def score_pair(output, pair, raw, annotation):
    reader, tokenizer = CloudReader(), BudgetTokenizer()
    q1, _, docs1 = adapt(raw[pair['first']], 'musique', pair['family_id'])
    q, label, docs = adapt(raw[pair['second']], 'musique', pair['family_id'])
    warm, warm_rank = assemble(tokenizer, q1, docs1)
    start = perf_counter()
    ranked, rank = assemble(tokenizer, q, docs)
    rank_wall = (perf_counter() - start) * 1000
    full = pack(tokenizer, docs, [(i,s['index']) for i,d in enumerate(docs) for s in d['spans']])
    selected, hit = None, False
    if annotation['shared'] and annotation['new']:
        shared, new = annotation['shared'], annotation['new']
        assert not set(map(tuple, shared)) & set(map(tuple, new))
        assert all(docs[i]['title'] == pair['shared_title'] and docs[i]['text'] == pair['shared_text'] for i, _ in shared)
        selected = pack(tokenizer, docs, shared + new)
        assert selected['evidence_tokens'] <= BUDGET
        hit = store_shared(output / f'{pair["family_id"]}.sqlite3', docs1, warm, docs, shared)
    ready = hit and annotation['tier'] == 'clear'
    available = {'raw_full':full, 'full_rerank':ranked, 'warm_view_route':selected if ready else ranked}
    if selected:
        available['manual_short'] = selected
    order = random.Random(q['id']).sample(sorted(available), len(available))
    rows = read_question(reader, [request(q, docs, m, available[m]) for m in order], {})
    scored = []
    for row in rows:
        ranked_online = row['method']=='full_rerank' or (row['method']=='warm_view_route' and not ready)
        scored.append(score_row({**row, 'warm_hit':hit, 'eligible':ready, 'rank_calls':int(ranked_online),
                                 'rank_tokens':rank['api_prompt_tokens'] if ranked_online else 0,
                                 'rank_cost':rank['list_price_cny'] if ranked_online else 0,
                                 'rank_wall_ms':rank_wall if ranked_online else 0}, q, {d['id']:d for d in docs}, label))
    log = {'pair':pair, 'annotation':annotation, 'warm_rank':warm_rank, 'full_rank':rank,
           'warm_selected':warm['selected_blocks'], 'warm_hit':hit, 'eligible':ready}
    return scored, log


def run(output: Path) -> None:
    if (output / 'scored.jsonl').exists():
        raise FileExistsError(output / 'scored.jsonl')
    frozen = json.loads((output / 'freeze.json').read_text(encoding='utf-8'))
    annotations = json.loads((output / 'audit.json').read_text(encoding='utf-8'))
    assert {p['second'] for p in frozen['pairs']} == set(annotations)
    source = Path(__file__).read_bytes()
    (output / 'source.py').write_bytes(source)
    dump(output / 'config.json', {'reader':MODEL, 'reranker':RERANK_MODEL, 'budget':BUDGET,
                                 'source_sha256':hashlib.sha256(source).hexdigest(),
                                 'audit_sha256':hashlib.sha256((output/'audit.json').read_bytes()).hexdigest()})
    raw = {r['id']:r for r in raw_rows('musique')}
    rows, logs = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for scored, log in pool.map(lambda pair: score_pair(output, pair, raw, annotations[pair['second']]), frozen['pairs']):
            rows.extend(scored)
            logs.append(log)
            print(log['pair']['second'], log['warm_hit'], [(r['method'], round(r['f1'],3)) for r in scored], flush=True)
    jsonl(output / 'scored.jsonl', rows)
    jsonl(output / 'warm.jsonl', logs)
    stats = {}
    for method in METHODS:
        group = [r for r in rows if r['method']==method]
        # Shared responses avoid charging the experiment twice; reconstruct single-arm costs from usage.
        from cloud_reader import INPUT_PRICE, OUTPUT_PRICE
        stats[method] = {'n':len(group), 'ok':sum(r['status']=='ok' for r in group), 'f1':mean(r['f1'] for r in group),
                         'em':mean(r['em'] for r in group), 'rank_calls':sum(r['rank_calls'] for r in group),
                         'rank_tokens':sum(r['rank_tokens'] for r in group),
                         'reader_tokens':sum(r.get('api_prompt_tokens',0) for r in group),
                         'evidence_p50':median(r['evidence_tokens'] for r in group),
                         'online_cost_cny':sum(r['rank_cost']+(INPUT_PRICE*r.get('api_prompt_tokens',0)+
                                               OUTPUT_PRICE*r.get('api_completion_tokens',0))/1e6 for r in group)}
    dump(output / 'summary.json', {'warm_hits':sum(log['warm_hit'] for log in logs),
                                   'eligible':sum(log['eligible'] for log in logs), 'methods':stats})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'inspect', 'run'])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'freeze':freeze, 'inspect':inspect, 'run':run}[args.stage](args.output)
