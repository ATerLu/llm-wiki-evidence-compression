"""Measure exact-query cache and versioned proof reuse on different natural questions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from statistics import mean, median
from time import perf_counter

from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, MODEL
from external_compression import RERANK_MODEL, dump, pack_ranked, read_question, rerank, sentence_candidates
from paired import score_row
from prepare import adapt, jsonl
from semantic_bridge_compile import raw_rows
from wiki_bridge import span_text

WORKLOAD = Path("runs/semantic-restoration-512/reuse-rescreen.json")
BUDGET = 1024
SPECS = {
    "Middle Ages": (("historia regum britanniae",), ("historia regum britanniae", "arthur")),
    "Biodiversity": (("prunus spinosa",), ("prunus spinosa", "belgium")),
    "Tammy Leitner": (("tammy leitner",), ("born", "san diego")),
    "Umayyad Caliphate": (("postal service", "umayyad"), ("mu'awiyah", "introduced postal service")),
    "Steven Spielberg": (("steven spielberg", "grandparents"), ("grandparents", "ukraine")),
}
METHODS = ("recompute", "exact_cache", "proof_view")


def assemble(tokenizer, query, docs, pin=()):
    choices = [item for item in sentence_candidates(docs) if item[:2] not in pin]
    rank = rerank(query['question'], [item[2] for item in choices])
    selected, text, count, _ = pack_ranked(tokenizer, docs, choices, rank['ranking'], BUDGET, pinned=pin)
    pack = {'selected_blocks': sorted((i, sorted(js)) for i, js in selected.items()),
            'evidence': text, 'evidence_tokens': count}
    return pack, rank


def hit_view(db, query, docs):
    sentence = query['question'].casefold()
    for terms, version, span in db.execute('SELECT terms, version, span FROM view_proof'):
        if all(term in sentence for term in json.loads(terms)):
            matches = [i for i, d in enumerate(docs) if d['version'] == version]
            if len(matches) == 1:
                return ((matches[0], span),)
    return ()


def write_warm(db, query, docs, pack, pair):
    db.execute('INSERT INTO evidence_cache VALUES (?,?)', (query['question'], json.dumps(pack)))
    trigger, proof_terms = SPECS[pair['shared_page']]
    eligible = [(i, j) for i, js in pack['selected_blocks'] for j in js
                if docs[i]['title'] == pair['shared_page'] and docs[i]['text'] == pair['source_text']
                and all(term in span_text(docs[i], j).casefold() for term in proof_terms)]
    if len(eligible) == 1 and all(term in query['question'].casefold() for term in trigger):
        i, j = eligible[0]
        db.execute('INSERT INTO view_proof VALUES (?,?,?)', (json.dumps(trigger), docs[i]['version'], j))
    db.commit()
    return bool(db.execute('SELECT count(*) FROM view_proof').fetchone()[0])


def evaluate_arm(db, reader, tokenizer, query, docs, label, workload, method):
    start = perf_counter()
    pin, cached = (), None
    if method == 'proof_view':
        pin = hit_view(db, query, docs)
    elif method == 'exact_cache':
        cached = db.execute('SELECT pack FROM evidence_cache WHERE question=?', (query['question'],)).fetchone()
    lookup_ms = 1000 * (perf_counter() - start)
    pack, rank = (json.loads(cached[0]), {}) if cached else assemble(tokenizer, query, docs, pin)
    row = {'dataset':'musique', 'query_id':query['id'], 'family_id':query['family_id'], 'question':query['question'],
           'candidate_ids':[d['id'] for d in docs], 'budget':BUDGET, 'workload':workload, 'method':method,
           'lookup_ms':lookup_ms, 'rank_tokens':rank.get('api_prompt_tokens', 0),
           'rank_cost':rank.get('list_price_cny', 0), 'rank_api_ms':rank.get('api_ms', 0),
           'rank_response_id':rank.get('api_response_id'), 'pinned':pin, 'cache_hit':bool(cached), **pack}
    row = read_question(reader, [row], {})[0]
    row['wall_ms'] = 1000 * (perf_counter() - start)
    return score_row(row, query, {d['id']:d for d in docs}, label)


def run_pair(output, pair, raw):
    q1, q2 = [adapt(raw[qid], 'musique', pair['family_id']) for qid in pair['ids']]
    reader, tokenizer = CloudReader(), BudgetTokenizer()
    with closing(sqlite3.connect(output / f"{pair['family_id']}.sqlite3")) as db:
        db.executescript('CREATE TABLE evidence_cache (question TEXT PRIMARY KEY, pack TEXT);'
                         'CREATE TABLE view_proof (terms TEXT, version TEXT, span INTEGER);')
        warm_pack, rank = assemble(tokenizer, q1[0], q1[2])
        wrote = write_warm(db, q1[0], q1[2], warm_pack, pair)
        if wrote:
            assert hit_view(db, q1[0], q1[2])
            altered = [{**d, 'version': 'different'} for d in q1[2]]
            assert not hit_view(db, q1[0], altered)
        rng = random.Random(pair['family_id'])
        rows = [evaluate_arm(db, reader, tokenizer, q, docs, label, workload, method)
                for workload, (q, label, docs) in (('related_new', q2), ('exact_repeat', q1))
                for method in rng.sample(METHODS, len(METHODS))]
    return rows, {'family':pair['family_id'], 'view_seeded':wrote, 'warm_rank':rank}


def run(output):
    output.mkdir(parents=True)
    pairs = [x for x in json.loads(WORKLOAD.read_text(encoding='utf-8'))['sample'] if x['shared_page'] in SPECS]
    source = Path(__file__).read_bytes()
    (output / 'source.py').write_bytes(source)
    dump(output / 'config.json', {'model':MODEL, 'rerank_model':RERANK_MODEL, 'budget':BUDGET, 'specs':SPECS,
                                 'pairs':[p['ids'] for p in pairs], 'source_sha256':hashlib.sha256(source).hexdigest()})
    raw = {r['id']:r for r in raw_rows('musique')}
    rows, warm = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for group, log in pool.map(lambda pair: run_pair(output, pair, raw), pairs):
            rows.extend(group)
            warm.append(log)
            print(log['family'], log['view_seeded'], flush=True)
    jsonl(output / 'scored.jsonl', rows)
    jsonl(output / 'warm.jsonl', warm)
    summary = {'pairs':len(pairs), 'warm_views':sum(x['view_seeded'] for x in warm), 'results':{}}
    for workload in ('related_new', 'exact_repeat'):
        summary['results'][workload] = {}
        for method in METHODS:
            group = [r for r in rows if (r['workload'], r['method']) == (workload, method)]
            summary['results'][workload][method] = {
                'ok':sum(r['status']=='ok' for r in group), 'f1':mean(r['f1'] for r in group),
                'rank_tokens':sum(r['rank_tokens'] for r in group), 'rank_calls':sum(bool(r['rank_response_id']) for r in group),
                'hits':sum(bool(r['pinned']) or r['cache_hit'] for r in group), 'wall_p50_ms':median(r['wall_ms'] for r in group),
                'reader_tokens':sum(r.get('api_prompt_tokens', 0) for r in group),
                'rank_and_reader_cost_cny':sum(r['rank_cost'] + r.get('list_price_cny', 0) for r in group),
            }
    dump(output / 'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
