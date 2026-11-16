"""Prepare and run a controlled 512-token evidence restoration diagnostic."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random

from baseline import sentence_blocks
from cloud_experiment import BudgetTokenizer
from cloud_reader import CloudReader, MODEL
from external_compression import dump, read_question
from paired import score_row
from prepare import jsonl, read_jsonl
from wiki_bridge import span_text

SOURCE = Path("runs/semantic-bridge-heldout100")
BUDGET = 512
LENGTH_TOLERANCE = 3


def data() -> tuple[dict, dict, dict, dict, dict]:
    queries = {r['id']: r for r in read_jsonl(SOURCE / 'queries.jsonl') if r['dataset'] == 'musique'}
    corpus = {r['id']: r for r in read_jsonl(SOURCE / 'corpus.jsonl')}
    labels = {r['id']: r for r in read_jsonl(SOURCE / 'labels-musique.jsonl')}
    ranks = {r['query_id']: r for r in read_jsonl(SOURCE / 'rankings.jsonl') if r['dataset'] == 'musique'}
    previous = {(r['query_id'], r['method'], r['budget']): r for r in read_jsonl(SOURCE / 'scored.jsonl')
                if r['dataset'] == 'musique'}
    return queries, corpus, labels, ranks, previous


def render(tokenizer, docs: list[dict], pairs: set[tuple]) -> tuple[dict, str, int]:
    selected = {}
    for i, j in sorted(pairs):
        selected.setdefault(i, set()).add(j)
    text = sentence_blocks(docs, selected)
    return selected, text, len(tokenizer.tokenize(text))


def choose_control(tokenizer, docs, kept, choices, target_size):
    # Standalone span lengths screen options; rendering then includes titles and exact joins.
    titles = {i for i, _ in kept}
    approx = {p: len(tokenizer.tokenize(span_text(docs[p[0]], p[1]))) +
              (0 if p[0] in titles else len(tokenizer.tokenize(f"[{docs[p[0]]['title']}]\n"))) for p in choices}
    base_size = render(tokenizer, docs, kept)[2]
    options = [(p,) for p in choices] + list(combinations(choices, 2))
    options.sort(key=lambda spans: (abs(sum(approx[p] for p in spans) + base_size - target_size), len(spans), spans))
    for spans in options:
        size = render(tokenizer, docs, kept | set(spans))[2]
        if size <= BUDGET and abs(size - target_size) <= LENGTH_TOLERANCE:
            return set(spans)
    raise ValueError(f'No control within {LENGTH_TOLERANCE} tokens of {target_size}')


def plan_case(tokenizer, docs, query, label, rank, base, annotation):
    selected = {(i, j) for i, ids in base['selected_blocks'] for j in ids}
    restore = {tuple(p) for p in annotation['restore']}
    assert restore and not (restore & selected)
    refs = {i for i, c in enumerate(query['candidates']) if c['key'] in label['support']}
    ordering = [tuple(rank['candidate_spans'][i]) for i in rank['ranking']]
    kept = selected.copy()
    removal_order = [p for p in reversed(ordering) if p in selected and p[0] not in refs]
    for pair in removal_order:
        if render(tokenizer, docs, kept | restore)[2] <= BUDGET:
            break
        kept.remove(pair)
    target_size = render(tokenizer, docs, kept | restore)[2]
    if target_size > BUDGET:
        raise ValueError(f"Cannot fit restoration without deleting an existing reference: {query['id']}")
    choices = sorted(p for p in ordering if p[0] not in refs and p not in (selected | restore))
    control = (set(map(tuple, annotation['control'])) if 'control' in annotation else
               choose_control(tokenizer, docs, kept, choices, target_size))
    assert control and control <= set(choices)
    variants = {'original': selected, 'restore': kept | restore, 'control': kept | control}
    requests = []
    for method, pairs in variants.items():
        blocks, evidence, size = render(tokenizer, docs, pairs)
        assert size <= BUDGET
        if method == 'original':
            assert evidence == base['evidence']
        requests.append({**{k: base[k] for k in ('dataset', 'query_id', 'family_id', 'candidate_ids', 'question')},
                         'method': method, 'budget': BUDGET, 'selected_blocks': sorted((i, sorted(ids)) for i, ids in blocks.items()),
                         'evidence': evidence, 'evidence_tokens': size})
    assert abs(requests[1]['evidence_tokens'] - requests[2]['evidence_tokens']) <= LENGTH_TOLERANCE
    excerpts = {name: [{'doc': i, 'span': j, 'title': docs[i]['title'], 'version': docs[i]['version'],
                       'text': span_text(docs[i], j)} for i, j in sorted(spans)] for name, spans in
                (('restored', restore), ('control', control), ('removed', selected - kept))}
    return {'query_id': query['id'], 'tier': annotation['tier'], **excerpts, 'requests': requests}


def prepare(output: Path, annotations: dict) -> None:
    queries, corpus, labels, ranks, previous = data()
    tokenizer = BudgetTokenizer()
    missed = sorted(q for q in queries if not previous[q, 'semantic', BUDGET]['all_support_touched'])
    touched = sorted(q for q in queries if previous[q, 'semantic', BUDGET]['all_support_touched']
                     and not previous[q, 'semantic', BUDGET]['em'])
    checked = random.Random(20261118).sample(touched, 5)
    assert set(annotations) == set(missed + checked)
    cases = []
    for q in sorted(annotations):
        annotation = annotations[q]
        if annotation.get('restore'):
            query = queries[q]
            docs = [corpus[c['id']] for c in query['candidates']]
            cases.append(plan_case(tokenizer, docs, query, labels[q], ranks[q], previous[q, 'semantic', BUDGET], annotation))
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (output / 'source.py').write_bytes(source)
    dump(output / 'plan.json', {'model': MODEL, 'budget': BUDGET, 'length_tolerance': LENGTH_TOLERANCE,
                               'source_sha256': hashlib.sha256(source).hexdigest(),
                               'missed_ids': missed, 'touched_sample_ids': checked, 'annotations': annotations, 'cases': cases})


def run(output: Path) -> None:
    if (output / 'scored.jsonl').exists():
        raise FileExistsError(output / 'scored.jsonl')
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    assert plan['model'] == MODEL and plan['budget'] == BUDGET
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['source_sha256']
    queries, corpus, labels, _, previous = data()
    reader, rng = CloudReader(), random.Random(20261119)
    scored = []
    for case in plan['cases']:
        qid = case['query_id']
        requests = case['requests']
        rng.shuffle(requests)
        for row in read_question(reader, requests, {}):
            scored.append(score_row(row, queries[qid], corpus, labels[qid]))
        print(qid, [(r['method'], round(r['f1'], 3), r['status']) for r in scored[-3:]], flush=True)
    jsonl(output / 'scored.jsonl', scored)
    summary = {}
    for case in plan['cases']:
        qid = case['query_id']
        summary[qid] = {'tier': case['tier'], 'gold': labels[qid]['answer'],
                        'historical_original_f1': previous[qid, 'semantic', BUDGET]['f1'],
                        'conditions': {r['method']: {k: r[k] for k in ('answer', 'em', 'f1', 'status', 'evidence_tokens')}
                                       for r in scored if r['query_id'] == qid}}
    dump(output / 'summary.json', {'model': MODEL, 'answer_calls': len(scored),
                                   'cost_cny': sum(r.get('list_price_cny', 0) for r in scored), 'cases': summary})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    if args.annotations:
        prepare(args.output, json.loads(args.annotations.read_text(encoding='utf-8')))
    else:
        run(args.output)
