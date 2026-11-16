"""Replay two audited same-author cases from archived ranks and raw compilation responses."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from unittest.mock import patch

from author_bridge import asks_work, author_source
from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, MODEL
from external_compression import dump, read_question, sentence_candidates
from paired import score_row
from prepare import adapt, jsonl, question_groups, read_jsonl
from semantic_bridge_compile import parse_batch, prior_ids, raw_rows
from semantic_bridge_experiment import packs
from skip_rerank_diagnostic import pack, request
from wiki_bridge import span_text, title_surface
from wiki_compile_v2 import fold, subject_anchors, work_author_phrase

BUDGETS = (512,1024)
NAME = re.compile(r'\bby\s+(?:the (?:[\w-]+ ){0,4}(?:author|writer|playwright|philosopher|novelist),\s*)?'
                  r"([A-Z][\w.\u2019'-]+(?:\s+[A-Z][\w.\u2019'-]+){1,4})")


def selected(row):
    return {(i,j) for i, js in row['selected_blocks'] for j in js}


def step_hits(annotation, spans):
    return [any(set(map(tuple, proof)) <= spans for proof in s['alternatives']) for s in annotation['steps']]


def load_case(qid, run):
    path = Path('runs')/run
    q = next(r for r in read_jsonl(path/'queries.jsonl') if r['id']==qid)
    corpus = {r['id']:r for r in read_jsonl(path/'corpus.jsonl')}
    docs = [corpus[x['id']] for x in q['candidates']]
    label = next(r for r in read_jsonl(path/'labels-musique.jsonl') if r['id']==qid)
    old = {(r['method'],r['budget']):r for r in read_jsonl(path/'scored.jsonl') if r['query_id']==qid}
    return path,q,docs,label,old,corpus


def compile_from_saved(path, docs, corpus):
    ids = {d['id'] for d in docs}
    rows = {r['doc_id']:r for api in read_jsonl(path/'compile-api.jsonl') if ids.intersection(api['pages'])
            for r in parse_batch(api,[corpus[i] for i in api['pages']],author_extension=True)}
    assert all(rows[d['id']]['parse']=='ok' for d in docs)
    return [rows[d['id']]['llm'] for d in docs]


def audit_case(q, docs, label, old, annotation):
    ref_keys = set(label['support'])
    steps = []
    for s in annotation['steps']:
        variants = [[{'candidate':i,'span':j,'source':docs[i]['title'],'version':docs[i]['version'],
                      'reference_paragraph':q['candidates'][i]['key'] in ref_keys,'text':span_text(docs[i],j)}
                     for i,j in option] for option in s['alternatives']]
        steps.append({'fact':s['fact'],'alternatives':variants})
    return {'query_id':q['id'],'question':q['question'],'tier':annotation['tier'],'note':annotation['note'],'steps':steps,
            'old_semantic_step_hits':{b:step_hits(annotation,selected(old['semantic',b])) for b in BUDGETS}}


def author_replay(tokenizer, path, q, docs, old, corpus, annotation):
    archived_rank = next(r for r in read_jsonl(path/'rankings.jsonl') if r['query_id']==q['id'])
    rank = {k:v for k,v in archived_rank.items() if k.startswith('api_') or k in
            {'status','ranking','scores','candidate_spans','list_price_cny'}}
    facts = compile_from_saved(path,docs,corpus)
    with patch('semantic_bridge_experiment.score_candidates',return_value=(sentence_candidates(docs),rank)):
        prepared, log = packs(tokenizer,q,docs,facts,author_extension=True)
    assert log['same_author_candidate']
    requests, diffs = [], []
    for b in BUDGETS:
        pure, rule, new = (prepared[m,b] for m in ('semantic','semantic_rule','semantic_wiki'))
        assert all(r['evidence']==old[r['method'],b]['evidence'] for r in (pure,rule))
        added, removed = selected(new)-selected(rule), selected(rule)-selected(new)
        diffs.append({'budget':b,'added':sorted(added),'removed':sorted(removed),
                      'before_steps':step_hits(annotation,selected(rule)), 'after_steps':step_hits(annotation,selected(new)),
                      'author_events':[x for x in new['wiki_edges'] if x.get('link_type')=='same_author']})
        requests.extend((pure,rule,new))
        if b==512:
            requests.extend(probes(tokenizer,q,docs,new,annotation))
    return [prepared['raw_full',16000],*requests],diffs


def probes(tokenizer,q,docs,new,annotation):
    original = selected(new)
    first = set(map(tuple,annotation['steps'][0]['alternatives'][0]))
    assert first <= original
    requests = [with_budget(q,docs,'drop_first',pack(tokenizer,docs,original-first))]
    if 'contrast' in annotation:
        target = set(map(tuple,annotation['steps'][1]['alternatives'][0]))
        substitute = set(map(tuple,annotation['contrast']))
        assert target <= original and not substitute.intersection(original)
        contrast = pack(tokenizer,docs,(original-target)|substitute)
        assert abs(contrast['evidence_tokens']-new['evidence_tokens'])<=3
        requests.append(with_budget(q,docs,'different_author',contrast))
    return requests


def with_budget(q,docs,method,evidence):
    assert evidence['evidence_tokens']<=512
    return {**request(q,docs,method,evidence),'budget':512}


def screen():
    results = {}
    for dataset in ('musique','hotpot'):
        rows = raw_rows(dataset)
        used = prior_ids(dataset)
        family = {q:f for f,ids in question_groups(rows,dataset).items() if not used.intersection(ids) for q in ids}
        focus = [r for r in rows if r['id'] in family and asks_work(r['question'])]
        candidates = []
        for raw in focus:
            q,_,docs = adapt(raw,dataset,family[raw['id']])
            titles = {d['id']:title_surface(d['title']) for d in docs}
            source = author_source(q['question'],docs,titles)
            if source is None:
                continue
            authors = [source_author_names(d) for d in docs]
            targets = [d['title'] for i,d in enumerate(docs) if i!=source and authors[i]&authors[source]
                       and fold(titles[d['id']])!=fold(titles[docs[source]['id']])]
            if targets:
                candidates.append({'id':q['id'],'family_id':q['family_id'],'targets':targets})
        results[dataset] = {'work_question_cue':len(focus),'matching_source_pairs':candidates}
    return results


def source_author_names(doc):
    names = set()
    for i in subject_anchors(doc):
        text = span_text(doc,i)
        for hit in NAME.finditer(text):
            candidate = hit[1].rstrip('.')
            if work_author_phrase(text,candidate):
                names.add(fold(candidate))
    return names


def run(output):
    if (output/'scored.jsonl').exists():
        raise FileExistsError(output/'scored.jsonl')
    annotations = json.loads((output/'audit.json').read_text(encoding='utf-8'))
    sources = ('author_dependency_replay.py','author_bridge.py','wiki_compile_v2.py','semantic_bridge_experiment.py',
               'semantic_bridge_compile.py','external_compression.py','cloud_reader.py')
    snapshot = output/'source'
    snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes(Path(name).read_bytes())
    dump(output/'config.json',{'model':MODEL,'budgets':BUDGETS,'audit_sha256':hashlib.sha256((output/'audit.json').read_bytes()).hexdigest(),
                               'source_sha256':{name:hashlib.sha256((snapshot/name).read_bytes()).hexdigest() for name in sources}})
    dump(output/'fresh-screen.json',screen())
    reader, tokenizer = CloudReader(), BudgetTokenizer()
    evidence, rows, differences = [], [], []
    for qid, annotation in annotations.items():
        path,q,docs,label,old,corpus = load_case(qid,annotation['run'])
        evidence.append(audit_case(q,docs,label,old,annotation))
        if not annotation.get('replay'):
            continue
        requests,diffs = author_replay(tokenizer,path,q,docs,old,corpus,annotation)
        differences.extend({'query_id':qid,**d} for d in diffs)
        random.Random(qid).shuffle(requests)
        rows.extend(score_row(r,q,corpus,label) for r in read_question(reader,requests,{}))
        print(qid,[(r['method'],r['budget'],round(r['f1'],3),r['status']) for r in rows if r['query_id']==qid],flush=True)
    jsonl(output/'evidence-audit.jsonl',evidence)
    jsonl(output/'scored.jsonl',rows)
    dump(output/'summary.json',{'audited':len(evidence),'logical_rows':len(rows),
                                'reader_calls':sum(not r['shared_api_result'] for r in rows),
                                'cost':sum(r.get('list_price_cny',0) for r in rows),'differences':differences})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
