"""Fixed Alibaba Cloud Model Studio reader with sanitized failures."""

import json
import os
import re
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL = "qwen3.5-plus-2026-04-20"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MAX_OUTPUT = 128
INPUT_PRICE = 0.8
OUTPUT_PRICE = 4.8


class ReaderError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CloudReader:
    def __init__(self) -> None:
        self._key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not self._key:
            raise ReaderError("DASHSCOPE_API_KEY is missing")
        self._opener = build_opener(NoRedirect())

    def generate(self, messages: list[dict], *, max_output: int = MAX_OUTPUT, json_object: bool = False) -> dict:
        payload = {
            "model": MODEL, "messages": messages, "temperature": 0,
            "max_tokens": max_output, "enable_thinking": False, "stream": False,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        request = Request(ENDPOINT, json.dumps(payload, ensure_ascii=False).encode("utf-8"), {
            "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
        })
        started = perf_counter()
        failures = []
        for attempt in range(1, 4):
            try:
                with self._opener.open(request, timeout=50) as response:
                    payload = json.load(response)
                return self._result(payload, started, attempt, failures)
            except HTTPError as error:
                with error:
                    try:
                        code = json.load(error).get("error", {}).get("code", "")
                    except (ValueError, AttributeError):
                        code = ""
                    code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else ""
                    failure = f"HTTP {error.code} {code}".strip()
                if error.code not in {429, 500, 502, 503, 504}:
                    raise ReaderError(failure) from None
            except (URLError, TimeoutError, ConnectionError) as error:
                failure = type(error).__name__
            except (ValueError, TypeError, KeyError, IndexError) as error:
                raise ReaderError(f"Invalid API response: {type(error).__name__}") from None
            failures.append(failure)
            if attempt < 3:
                sleep(attempt * 1.5)
        raise ReaderError("Retry limit: " + ", ".join(failures))

    def _result(self, payload: dict, started: float, attempt: int, failures: list[str]) -> dict:
        if payload["model"] != MODEL:
            raise ReaderError("API returned a different model")
        choice = payload["choices"][0]
        answer = (choice["message"].get("content") or "").strip()
        usage = payload["usage"]
        prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return {
            "answer": answer, "status": "ok" if answer else "empty_answer",
            "finish_reason": choice.get("finish_reason"), "api_attempts": attempt,
            "api_retry_statuses": failures, "api_response_id": payload.get("id"),
            "api_prompt_tokens": prompt, "api_completion_tokens": completion,
            "api_cached_tokens": cached, "api_ms": 1000 * (perf_counter() - started),
            "list_price_cny": (prompt * INPUT_PRICE + completion * OUTPUT_PRICE) / 1_000_000,
        }
