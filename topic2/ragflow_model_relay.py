"""Record model usage for an isolated RAGFlow experiment with fixed DashScope models."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import hmac
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from cloud_reader import ENDPOINT, MODEL, NoRedirect


EMBEDDING = "text-embedding-v4"
MODELS = {"/v1/chat/completions": MODEL, "/v1/embeddings": EMBEDDING}
UPSTREAM = ENDPOINT.rsplit("/v1/", 1)[0]
MAX_REQUEST_BYTES = 4 * 1024 * 1024
EMBED_BATCH = 10
WIKI_REWRITE = "You are the final editor for a knowledge-base Wiki page."


class RelayServer(ThreadingHTTPServer):
    request_queue_size = 128


class Relay(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def record(self, path, status, payload, started, rows, rewrite_digest):
        record = {"at": datetime.now(timezone.utc).isoformat(), "route": path, "model": MODELS[path],
                  "status": status, "ms": round((perf_counter() - started) * 1000, 2),
                  "usage": payload.get("usage"), "response_id": payload.get("id"),
                  "error_code": (payload.get("error") or {}).get("code"), "rows": rows}
        if rewrite_digest:
            record["wiki_rewrite_sha256"] = rewrite_digest
        with self.server.lock, self.server.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

    def upstream(self, path, body):
        started = perf_counter()
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        rewrite = any(WIKI_REWRITE in str(message.get("content", ""))[:500] for message in body.get("messages", []))
        request = Request(UPSTREAM + path, content, {
            "Authorization": "Bearer " + self.server.cloud_key, "Content-Type": "application/json",
        })
        try:
            with build_opener(NoRedirect()).open(request, timeout=570 if rewrite else 120) as response:
                payload = json.load(response)
            status = 200
        except HTTPError as error:
            with error:
                try:
                    upstream_error = json.load(error).get("error", {})
                    code = upstream_error.get("code")
                except (ValueError, TypeError, AttributeError):
                    code = None
            status = error.code
            payload = {"error": {"message": "DashScope rejected the request", "code": code}}
        except (URLError, TimeoutError, ConnectionError, HTTPException, ValueError):
            status, payload = 502, {"error": {"message": "DashScope upstream failure"}}
        inputs = body.get("input")
        self.record(path, status, payload, started, len(inputs) if isinstance(inputs, list) else None,
                    sha256(content).hexdigest() if rewrite else None)
        return status, payload

    def embeddings(self, path, body):
        inputs = body.get("input")
        if not isinstance(inputs, list) or len(inputs) <= EMBED_BATCH:
            return self.upstream(path, body)
        data, usage = [], {"prompt_tokens": 0, "total_tokens": 0}
        for start in range(0, len(inputs), EMBED_BATCH):
            batch = inputs[start:start + EMBED_BATCH]
            status, payload = self.upstream(path, {**body, "input": batch})
            if status != 200:
                return status, payload
            data.extend({**item, "index": item["index"] + start} for item in payload["data"])
            for key in usage:
                usage[key] += payload["usage"][key]
        return 200, {"object": "list", "model": EMBEDDING, "data": data, "usage": usage}

    def do_POST(self):
        path = self.path
        if self.client_address[0] not in self.server.allowed or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + self.server.local_token):
            return self.send_json(401, {"error": {"message": "Unauthorized"}})
        if path not in MODELS:
            return self.send_json(404, {"error": {"message": "Unknown endpoint"}})
        try:
            length = int(self.headers["Content-Length"])
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Invalid body length")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or body.get("model") != MODELS[path] or body.get("stream"):
                raise ValueError("Expected the pinned model with a non-streaming request")
        except (KeyError, TypeError, ValueError):
            return self.send_json(400, {"error": {"message": "Invalid experiment request"}})
        if path.endswith("/embeddings"):
            status, payload = self.embeddings(path, body)
        else:
            status, payload = self.upstream(path, {**body, "enable_thinking": False, "temperature": 0})
        self.send_json(status, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bind")
    parser.add_argument("allowed_wsl_address")
    parser.add_argument("token_file", type=Path)
    parser.add_argument("usage_log", type=Path)
    parser.add_argument("--port", type=int, default=17891)
    args = parser.parse_args()
    with RelayServer((args.bind, args.port), Relay) as server:
        server.allowed = {args.allowed_wsl_address, args.bind}
        server.cloud_key = os.environ["DASHSCOPE_API_KEY"].strip()
        server.local_token = args.token_file.read_text(encoding="utf-8").strip()
        server.log_path = args.usage_log
        server.lock = Lock()
        if not server.cloud_key or len(server.local_token) < 24:
            raise ValueError("Experiment credentials are unavailable")
        server.serve_forever()
