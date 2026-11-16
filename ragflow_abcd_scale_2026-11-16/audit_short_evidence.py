"""Check evidence sufficiency on the same related queries that failed automatic routing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median

from auto_view_experiment import lookup
from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, MODEL, INPUT_PRICE, OUTPUT_PRICE
from external_compression import dump, read_question
from paired import score_row
from prepare import adapt, jsonl
from semantic_bridge_compile import raw_rows
from skip_rerank_diagnostic import pack, request
from wiki_bridge import span_text

SOURCE = Path('runs/auto-view-heldout24')


def setup(tokenizer, pair, audit, raw):
    q, label, docs = adapt(raw[pair['second']], 'musique', pair['family_id'])
    full = pack(tokenizer, docs, [(i,s['index']) for i,d in enumerate(docs) for s in d['spans']])
    requests = [request(q,docs,'raw_full',full)]
    records = {(r['version'],r['span']):r['text'] for r in lookup(SOURCE/'evidence.sqlite3',docs)
               if r['qid']==pair['first']}
    detail = {'query_id':q['id'], 'family_id':q['family_id'], 'warm_query_id':pair['first'],
              'question':q['question'], **audit, 'spans':[]}
    if audit['shared']:
        shared = list(map(tuple,audit['shared']))
        rest = list(map(tuple,audit['rest']))
        assert rest and len(set(shared+rest))==len(shared+rest)
        assert all(docs[i]['title']==pair['shared_title'] and docs[i]['text']==pair['shared_source_text'] for i,_ in shared)
        for kind, spans in (('shared',shared),('rest',rest)):
            for i,j in spans:
                doc = docs[i]
                text = span_text(doc,j)
                detail['spans'].append({'role':kind,'candidate':i,'span':j,'title':doc['title'],'version':doc['version'],
                                        'text':text,'from_warm':records.get((doc['version'],j))==text})
        detail['shared_ready'] = all(s['from_warm'] for s in detail['spans'] if s['role']=='shared')
        requests.append(request(q,docs,'manual_short',pack(tokenizer,docs,shared+rest)))
        if audit['tier']=='clear':
            requests.append(request(q,docs,'without_shared',pack(tokenizer,docs,rest)))
    else:
        assert audit['tier']=='blocked' and not audit['rest']
        detail['shared_ready'] = False
    return q,label,docs,detail,requests


def evaluate(job):
    q,label,docs,detail,requests = job
    random.Random(q['id']).shuffle(requests)
    corpus = {d['id']:d for d in docs}
    rows = [score_row({**row, 'tier':detail['tier']},q,corpus,label)
            for row in read_question(CloudReader(),requests,{})]
    print(q['id'],detail['tier'],[(r['method'],r['status'],round(r['f1'],3)) for r in rows],flush=True)
    return rows


def summary(rows, details):
    groups = {r['query_id']:{x['method']:x for x in rows if x['query_id']==r['query_id']} for r in details}
    out = {'cases':len(details), 'api_calls':sum(not r['shared_api_result'] for r in rows),
           'successful_rows':sum(r['status']=='ok' for r in rows),
           'actual_cost_cny':sum(r.get('list_price_cny',0) for r in rows), 'tiers':{}}
    for tier in ('clear','qualified','blocked'):
        cases = [r for r in details if r['tier']==tier]
        table = {'n':len(cases), 'shared_ready':sum(r['shared_ready'] for r in cases), 'methods':{}}
        for method in ('raw_full','manual_short','without_shared'):
            selected = [groups[r['query_id']][method] for r in cases if method in groups[r['query_id']]]
            if selected:
                table['methods'][method] = {'n':len(selected),'f1':mean(r['f1'] for r in selected),
                    'em':mean(r['em'] for r in selected),'evidence_p50':median(r['evidence_tokens'] for r in selected),
                    'reader_input':sum(r.get('api_prompt_tokens',0) for r in selected),
                    'reader_cost_cny':sum((r.get('api_prompt_tokens',0)*INPUT_PRICE+
                                          r.get('api_completion_tokens',0)*OUTPUT_PRICE)/1e6 for r in selected)}
        out['tiers'][tier]=table
    return out


def run(output):
    if (output/'scored.jsonl').exists():
        raise FileExistsError(output/'scored.jsonl')
    pairs = json.loads((SOURCE/'freeze.json').read_text(encoding='utf-8'))['pairs']
    audits = json.loads((output/'audit.json').read_text(encoding='utf-8'))
    assert set(audits)=={p['second'] for p in pairs}
    raw = {r['id']:r for r in raw_rows('musique')}
    t = BudgetTokenizer()
    jobs = [setup(t,p,audits[p['second']],raw) for p in pairs]
    details = [x[3] for x in jobs]
    script = Path(__file__).read_bytes()
    (output/'source.py').write_bytes(script)
    dump(output/'config.json',{'model':MODEL,'source_run':str(SOURCE),'source_sha256':hashlib.sha256(script).hexdigest(),
                               'audit_sha256':hashlib.sha256((output/'audit.json').read_bytes()).hexdigest(),
                               'warm_db_sha256':hashlib.sha256((SOURCE/'evidence.sqlite3').read_bytes()).hexdigest()})
    jsonl(output/'evidence-audit.jsonl',details)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for group in pool.map(evaluate,jobs) for r in group]
    jsonl(output/'scored.jsonl',rows)
    dump(output/'summary.json',summary(rows,details))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
