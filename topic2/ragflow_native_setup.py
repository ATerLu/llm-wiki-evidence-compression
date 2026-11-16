"""Provision the isolated RAGFlow v1.0 source and native Wiki pipeline."""

import argparse
import base64
import json
from pathlib import Path
import secrets
import subprocess
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from cloud_reader import MODEL, NoRedirect
from dependency_development import SOURCE
from external_compression import dump
from ragflow_model_relay import EMBEDDING
from semantic_bridge_experiment import verify_freeze


PLATFORM = Path("platform/ragflow-1.0")
RELEASE = PLATFORM / "release"
API = "http://127.0.0.1:19380/api/v1"
PROVIDER = "OpenAI-API-Compatible"
INSTANCE = "topic2-dashscope-relay"
ACCOUNT = "topic2-ragflow@example.com"
NAMES = {"pilot": "topic2-native-wiki-pilot", "full": "topic2-native-wiki-100"}


class Ragflow:
    def __init__(self, token=""):
        self.token = token
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, method, path, body=None, *, params=None, files=None, error_ok=False):
        url = API + path + ("?" + urlencode(params) if params else "")
        headers = {"Authorization": self.token} if self.token else {}
        data = None
        if files:
            boundary = "ragflow" + secrets.token_hex(16)
            parts = []
            for name, content in files:
                parts.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                              f"filename=\"{name}\"\r\nContent-Type: text/plain\r\n\r\n".encode(), content, b"\r\n"])
            data = b"".join([*parts, f"--{boundary}--\r\n".encode()])
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            with self.opener.open(Request(url, data, headers, method=method), timeout=120) as response:
                payload, response_headers = json.load(response), response.headers
        except HTTPError as error:
            with error:
                try:
                    payload, response_headers = json.load(error), error.headers
                except ValueError:
                    raise RuntimeError(f"{method} {path}: HTTP {error.code}") from None
        if error_ok:
            return payload, response_headers
        if payload.get("code") != 0:
            raise RuntimeError(f"{method} {path}: {payload.get('code')} {payload.get('message')}")
        return payload.get("data")


def authenticate():
    login_file = PLATFORM / "ragflow-login.local"
    if login_file.exists():
        client = Ragflow(login_file.read_text(encoding="ascii").strip())
        existing, _ = client.call("GET", "/users/me", error_ok=True)
        if existing["code"] == 0:
            return client
    password_file = PLATFORM / "ragflow-password.local"
    try:
        with password_file.open("x", encoding="ascii") as handle:
            handle.write("RagFlow-" + secrets.token_urlsafe(30) + "-7")
    except FileExistsError:
        pass
    encrypted = subprocess.run(
        ["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(RELEASE / "conf/public.pem"),
         "-pkeyopt", "rsa_padding_mode:pkcs1"],
        input=base64.b64encode(password_file.read_bytes()), check=True, capture_output=True,
    ).stdout
    payload = {"email": ACCOUNT, "password": base64.b64encode(encrypted).decode("ascii")}
    client = Ragflow()
    login, headers = client.call("POST", "/auth/login", payload, error_ok=True)
    if login["code"] != 0:
        client.call("POST", "/users", {**payload, "nickname": "Topic 2 Experiment"})
        login, headers = client.call("POST", "/auth/login", payload, error_ok=True)
    if login["code"] != 0 or not headers.get("Authorization"):
        raise RuntimeError(f"Unable to authenticate locally: {login.get('message')}")
    login_file.write_text(headers["Authorization"], encoding="ascii")
    return Ragflow(headers["Authorization"])


def collection(payload, *keys):
    if isinstance(payload, list):
        return payload
    for key in keys:
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Expected a collection with one of these keys: {keys}")


def models(client, relay):
    path = f"/providers/{PROVIDER}/instances"
    providers = collection(client.call("GET", "/providers"), "providers")
    if not any(p.get("name", p.get("provider_name")) == PROVIDER for p in providers):
        client.call("PUT", "/providers", {"provider_name": PROVIDER})
    instances = collection(client.call("GET", path), "instances")
    if not any(p.get("instance_name") == INSTANCE for p in instances):
        client.call("POST", path, {"instance_name": INSTANCE, "region": "default", "base_url": relay,
                                   "api_key": (PLATFORM / "relay-token.local").read_text().strip()})
    path += f"/{INSTANCE}/models"
    existing = collection(client.call("GET", path), "models")
    for model, kind, limit in ((MODEL, "chat", 32768), (EMBEDDING, "embedding", 8192)):
        if not any(row.get("model_name", row.get("name")) == model for row in existing):
            client.call("POST", path, {"model_name": model, "model_type": [kind], "max_tokens": limit})
        client.call("PATCH", "/models/default", {"model_provider": PROVIDER, "model_instance": INSTANCE,
                                                "model_name": model, "model_type": kind})
    model_rows = collection(client.call("GET", "/models/default"), "models")
    found = {r["model_type"]: (r.get("model_id") or r["id"]) for r in model_rows
             if r.get("model_name", r.get("name")) in (MODEL, EMBEDDING)}
    if set(found) != {"chat", "embedding"}:
        raise RuntimeError("Could not resolve both configured tenant model IDs")
    return found


def pipeline_dsl(group_id, model_id):
    general = json.loads((RELEASE / "internal/ingestion/pipeline/template/ingestion_pipeline_general.json")
                         .read_text(encoding="utf-8"))["dsl"]
    compiled = json.loads((RELEASE / "internal/agent/templates/compiler.json").read_text(encoding="utf-8"))
    nodes = general["components"]
    compiler_id, compiler = next((k, v) for k, v in compiled["components"].items() if k.startswith("Compiler:"))
    indexer_id = next(k for k, v in nodes.items() if v["obj"]["component_name"] == "Tokenizer")
    indexer = nodes[indexer_id]
    upstream = indexer["upstream"]
    compiler.update(upstream=upstream, downstream=[indexer_id])
    fields = {"compilation_template_group_id": group_id, "llm_id": model_id}
    compiler["obj"]["params"].update(fields)
    for k in upstream:
        node = nodes[k]
        node["downstream"] = [compiler_id if v == indexer_id else v for v in node["downstream"]]
    indexer["upstream"] = [compiler_id]
    nodes[compiler_id] = compiler
    graph = general["graph"]
    canvas_node = next(n for n in compiled["graph"]["nodes"] if n["id"] == compiler_id)
    canvas_node["data"]["form"].update(fields)
    graph["nodes"].append(canvas_node)
    for edge in graph["edges"]:
        if edge["target"] == indexer_id and edge["source"] in upstream:
            edge.update(target=compiler_id, id=f"{edge['source']}-{compiler_id}")
    graph["edges"].append({"source": compiler_id, "target": indexer_id, "id": f"{compiler_id}-{indexer_id}",
                           "sourceHandle": "start", "targetHandle": "end"})
    return general


def provision(client, output, ids):
    state_path = output / "native-pipeline.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    builtins = collection(client.call("GET", "/compilation-templates/builtins"), "templates")
    template = next(r for r in builtins if r.get("kind") == "wiki")
    group = client.call("POST", "/compilation-template-groups", {
        "name": "Topic 2 native Wiki",
        "templates": [{"name": "Wiki", "kind": template["kind"], "config": template["config"]}],
    })
    dsl = pipeline_dsl(group["id"], ids["chat"])
    agent = client.call("POST", "/agents", {"title": "Topic 2 native Wiki pipeline",
                                            "canvas_category": "dataflow_canvas", "dsl": dsl})
    version = client.call("POST", f"/agents/{agent['id']}/publish", {"dsl": dsl})
    state = {"group_id": group["id"], "pipeline_id": agent["id"], "pipeline_version_id": version["id"],
             "model_ids": ids, "dsl": dsl}
    dump(state_path, state)
    return state


def documents(client, dataset_id):
    page = 1
    while True:
        data = client.call("GET", f"/datasets/{dataset_id}/documents", params={"page": page, "page_size": 100})
        docs = collection(data, "docs", "documents")
        yield from docs
        if len(docs) < 100:
            return
        page += 1


def filenames(corpus):
    return {f"source-{index:04d}.txt": key for index, key in enumerate(sorted(corpus), 1)}


def ingest(client, output, scope, state):
    _, corpus, queries = verify_freeze(SOURCE)
    selected = queries if scope == "full" else [next(q for q in queries if q["dataset"] == name)
                                               for name in ("hotpot", "musique")]
    wanted = {r["id"] for q in selected for r in q["candidates"]}
    mapping = {name: key for name, key in filenames(corpus).items() if key in wanted}
    name = NAMES[scope]
    rows = collection(client.call("GET", "/datasets", params={"name": name}), "datasets")
    dataset = next((row for row in rows if row["name"] == name), None)
    parsing = {"parse_type": 2, "pipeline_id": state["pipeline_id"], "pipeline_version_id": state["pipeline_version_id"]}
    if dataset is None:
        dataset = client.call("POST", "/datasets",
                              {"name": name, "embedding_model": state["model_ids"]["embedding"], **parsing})
    dataset_id = dataset["id"]
    route = f"/datasets/{dataset_id}/documents"
    current = {r["name"]: r for r in documents(client, dataset_id)}
    pending = [(name, corpus[key]) for name, key in mapping.items() if name not in current]
    for index in range(0, len(pending), 50):
        files = [(name, (doc["title"] + "\n\n" + doc["text"]).encode("utf-8")) for name, doc in pending[index:index + 50]]
        client.call("POST", route, files=files)
    current = {r["name"]: r for r in documents(client, dataset_id)}
    if set(current) != set(mapping):
        raise RuntimeError("The uploaded document set differs from the frozen source")
    state = {"dataset_id": dataset_id, "questions": [q["id"] for q in selected],
             "documents": {key: current[name]["id"] for name, key in mapping.items()}}
    dump(output / f"native-{scope}.json", state)
    pending = [r["id"] for r in current.values() if r["ingestion_status"] in ("UNSTART", "FAILED", "CANCELED")]
    for index in range(0, len(pending), 80):
        client.call("POST", "/documents/ingest", {"doc_ids": pending[index:index + 80], "run": 1})
    print(f"{scope}: queued {len(pending)} of {len(current)} original documents", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("scope", choices=("pilot", "full"))
    parser.add_argument("--relay", default="http://172.18.96.1:17891/v1")
    args = parser.parse_args()
    client = authenticate()
    ids = models(client, args.relay)
    ingest(client, args.output, args.scope, provision(client, args.output, ids))
