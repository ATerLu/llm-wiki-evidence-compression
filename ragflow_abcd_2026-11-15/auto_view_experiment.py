"""Run warmed and no-view local routing on unused related and low-reuse families."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean, median
from time import perf_counter

from auto_view_route import choose_chain, cue_for
from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, MODEL
from external_compression import RERANK_MODEL, dump, read_question
from paired import score_row
from prepare import adapt, jsonl, question_groups, read_jsonl
from semantic_bridge_compile import prior_ids, raw_rows
from semantic_view_pilot import BUDGET, assemble
from skip_rerank_diagnostic import pack, request
from wiki_bridge import span_text

SEED = 20261123
N_RELATED = 12
N_LOW = 12
ARMS = ('raw_full', 'full_rerank', 'local_only', 'view_route')


def candidates_for_pairs(rows, families, excluded):
    by_source = defaultdict(list)
    for r in rows:
        family = families[r['id']]
        if family in excluded:
            continue
        first, *rest = r['question_decomposition']
        if re.search(r'#\d+', first['question']) or not any(re.search(r'#1\b', s['question']) for s in rest):
            continue
        shared = next(p for p in r['paragraphs'] if p['idx']==first['paragraph_support_idx'])
        by_source[family, str(first['id']), shared['title'], shared['paragraph_text']].append(r)
    pairs = defaultdict(dict)
    for (family, _, title, text), group in by_source.items():
        for a, b in combinations(sorted(group, key=lambda r:r['id']), 2):
            if a['question'] == b['question'] or a['answer'].strip().casefold()==b['answer'].strip().casefold():
                continue
            for first, second in ((a,b), (b,a)):
                pairs[family][first['id'], second['id']] = {
                    'family_id':family, 'first':first['id'], 'second':second['id'], 'shared_title':title,
                    'shared_source_text':text, 'supported_question_type':bool(cue_for(second['question'])),
                    'workload':'related'}
    return pairs


def freeze(output: Path):
    rows = raw_rows('musique')
    groups = question_groups(rows,'musique')
    families = {q:f for f, ids in groups.items() for q in ids}
    used = prior_ids('musique')
    previous = json.loads(Path('runs/skip-rerank-diagnostic12/freeze.json').read_text(encoding='utf-8'))
    excluded = {families[q] for q in used if q in families} | {p['family_id'] for p in previous['pairs']}
    pairs = candidates_for_pairs(rows, families, excluded)
    rng = random.Random(SEED)
    supported = sorted(f for f, choices in pairs.items() if any(p['supported_question_type'] for p in choices.values()))
    other = sorted(set(pairs)-set(supported))
    rng.shuffle(supported)
    rng.shuffle(other)
    related = []
    for f in (supported + other)[:N_RELATED]:
        preferred = [k for k,p in pairs[f].items() if p['supported_question_type']]
        related.append(pairs[f][rng.choice(sorted(preferred or pairs[f]))])
    low_families = sorted(groups.keys() - excluded - pairs.keys())
    raw = {r['id']:r for r in rows}
    # Include supported question types in the low-reuse workload when available.
    low_supported = [f for f in low_families if any(cue_for(raw[q]['question']) for q in groups[f])]
    low_other = sorted(set(low_families)-set(low_supported))
    rng.shuffle(low_supported)
    rng.shuffle(low_other)
    low = [{'family_id':f, 'second':rng.choice(sorted([q for q in groups[f] if cue_for(raw[q]['question'])] or groups[f])),
            'workload':'low_reuse'} for f in (low_supported+low_other)[:N_LOW]]
    assert len(related)==N_RELATED and len(low)==N_LOW
    output.mkdir(parents=True)
    dump(output/'freeze.json', {'seed':SEED, 'excluded_families':sorted(excluded), 'pairs':related,
                                'low_reuse':low, 'eligible_related_families':len(pairs),
                                'eligible_supported_related_families':len(supported)})


def warm_one(item, raw):
    q, _, docs = adapt(raw[item['first']], 'musique', item['family_id'])
    start = perf_counter()
    evidence, rank = assemble(BudgetTokenizer(), q, docs)
    records = [(q['id'], q['question'], docs[i]['version'], j, span_text(docs[i],j))
               for i, js in evidence['selected_blocks'] for j in js]
    return records, {'query_id':q['id'], 'rank_tokens':rank['api_prompt_tokens'], 'cost':rank['list_price_cny'],
                     'wall_ms':(perf_counter()-start)*1000, 'selected_spans':len(records)}


def build_view(output, items, raw):
    start = perf_counter()
    path = output/'evidence.sqlite3'
    with closing(sqlite3.connect(path)) as db, ThreadPoolExecutor(max_workers=4) as pool:
        db.executescript('CREATE TABLE proof (qid TEXT, question TEXT, version TEXT, span INTEGER, text TEXT);'
                         'CREATE INDEX proof_version ON proof(version);')
        logs = []
        for records, log in pool.map(lambda item:warm_one(item,raw), items):
            db.executemany('INSERT INTO proof VALUES (?,?,?,?,?)', records)
            logs.append(log)
        db.commit()
        assert db.execute('SELECT count(*) FROM proof WHERE version=?',('absent-source-version',)).fetchone()[0]==0
    jsonl(output/'warm.jsonl',logs)
    return {'build_wall_ms':(perf_counter()-start)*1000, 'rank_calls':len(logs),
            'rank_tokens':sum(r['rank_tokens'] for r in logs), 'cost':sum(r['cost'] for r in logs),
            'spans':sum(r['selected_spans'] for r in logs), 'size_bytes':path.stat().st_size}


def lookup(path, docs):
    versions = sorted({d['version'] for d in docs})
    placeholders = ','.join('?' for _ in versions)
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(f'SELECT * FROM proof WHERE version IN ({placeholders})',versions)]


def evaluate_one(output, item, raw):
    q, label, docs = adapt(raw[item['second']], 'musique', item['family_id'])
    tokenizer, reader = BudgetTokenizer(), CloudReader()
    corpus = {d['id']:d for d in docs}
    arms = random.Random(q['id']).sample(ARMS,len(ARMS))
    answers = []
    for method in arms:
        start = perf_counter()
        rank, decision = {}, {'reason':'baseline'}
        if method=='full_rerank':
            evidence, rank = assemble(tokenizer,q,docs)
        else:
            if method in ('local_only','view_route'):
                records = lookup(output/'evidence.sqlite3',docs) if method=='view_route' else []
                decision = choose_chain(q['question'],docs,records)
            if decision['reason']=='short':
                evidence = pack(tokenizer,docs,decision['spans'])
                if evidence['evidence_tokens'] > BUDGET:
                    decision['reason']='over_budget'
            if decision['reason']!='short':
                evidence = pack(tokenizer,docs,[(i,s['index']) for i,d in enumerate(docs) for s in d['spans']])
        req = request(q,docs,method if method=='full_rerank' or decision['reason']=='short' else 'raw_full',evidence)
        req['method'] = method
        row = read_question(reader,[req],{})[0]
        wall_ms = (perf_counter()-start)*1000
        answers.append(score_row({**row, 'workload':item['workload'], 'decision':decision,
                                  'rank_calls':int(bool(rank)), 'rank_tokens':rank.get('api_prompt_tokens',0),
                                  'rank_cost':rank.get('list_price_cny',0), 'wall_ms':wall_ms},q,corpus,label))
    reference = next(r for r in answers if r['method']=='raw_full')
    for row in answers:
        row['f1_shared_if_same_input'] = reference['f1'] if row['evidence']==reference['evidence'] else row['f1']
    return answers


def summarize(rows):
    stats = {}
    for workload in ('related','low_reuse','all'):
        stats[workload] = {}
        for method in ARMS:
            group = [r for r in rows if r['method']==method and (workload=='all' or r['workload']==workload)]
            stats[workload][method] = {'n':len(group), 'ok':sum(r['status']=='ok' for r in group),
                'short':sum(r['decision']['reason']=='short' for r in group),
                'used_view':sum(r['decision']['reason']=='short' and r['decision'].get('used_view',False) for r in group),
                'f1':mean(r['f1'] for r in group), 'f1_same_input_shared':mean(r['f1_shared_if_same_input'] for r in group),
                'rank_calls':sum(r['rank_calls'] for r in group), 'rank_tokens':sum(r['rank_tokens'] for r in group),
                'reader_tokens':sum(r.get('api_prompt_tokens',0) for r in group), 'p50_ms':median(r['wall_ms'] for r in group),
                'cost':sum(r['rank_cost']+r.get('list_price_cny',0) for r in group)}
    return stats


def run(output):
    if (output/'scored.jsonl').exists():
        raise FileExistsError(output/'scored.jsonl')
    frozen = json.loads((output/'freeze.json').read_text(encoding='utf-8'))
    items = frozen['pairs']+frozen['low_reuse']
    assert len({i['family_id'] for i in items})==len(items)
    assert not {i['family_id'] for i in items}.intersection(frozen['excluded_families'])
    files = ['auto_view_experiment.py','auto_view_route.py','semantic_view_pilot.py','skip_rerank_diagnostic.py',
             'cloud_reader.py','external_compression.py']
    source = output/'source'
    source.mkdir()
    for name in files:
        (source/name).write_bytes(Path(name).read_bytes())
    dump(output/'config.json',{'model':MODEL,'reranker':RERANK_MODEL,'budget':BUDGET,
                               'hashes':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}})
    raw = {r['id']:r for r in raw_rows('musique')}
    warmed = build_view(output,frozen['pairs'],raw)
    rng = random.Random(SEED)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, group in enumerate(pool.map(lambda item:evaluate_one(output,item,raw),rng.sample(items,len(items))),1):
            rows.extend(group)
            if i%4==0:
                print(f'{i}/{len(items)} queries',flush=True)
    jsonl(output/'scored.jsonl',rows)
    dump(output/'summary.json',{'warm':warmed,'online':summarize(rows)})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run'])
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.stage](args.output)
