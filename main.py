#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two API wires, ten free providers.

Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic),
JSON and SSE, plus GET /v1/models in the wire the auth header names. Each
request (model "auto") is routed to the strongest free model that meets its
requirements, speaking that provider's wire; a provider refusal is counted
and the request is answered by another eligible model. Registers its port and
catalog with the env, polls GET /work, and exits 0 when the workload is done.

Standard library only. Deterministic: every choice is a function of the
requests seen and the providers' answers; no wall clock, no unseeded
randomness.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The competition's fixed quality table (free models only), from the brief.
# Used only to rank models; eligibility (free/tools/vision/json/context)
# always comes from the live provider catalogs.
RATINGS = [
    ("groq", "llama-3.1-8b-instant", 1210),
    ("groq", "gpt-oss-120b", 1352),
    ("groq", "qwen3.8-27b", 1434),
    ("groq", "gpt-oss-20b", 1318),
    ("cerebras", "llama-4-scout-17b", 1250),
    ("cerebras", "qwen-3.8-27b", 1434),
    ("cerebras", "gpt-oss-120b", 1352),
    ("gemini", "gemini-3.8-flash", 1458),
    ("gemini", "gemma-4-31b", 1452),
    ("gemini", "gemma-4-26b", 1410),
    ("zai", "glm-4.7-flash", 1470),
    ("zai", "glm-4.6v-flash", 1432),
    ("openrouter", "z-ai/glm-5.3-flash:free", 1475),
    ("openrouter", "deepseek/deepseek-v4.1:free", 1462),
    ("openrouter", "minimax/m3:free", 1436),
    ("sambanova", "deepseek-v3.2", 1465),
    ("sambanova", "gemma-4-31b", 1452),
    ("pollinations", "mistral-nemo", 1180),
    ("pollinations", "openai-fast", 1390),
    ("cloudflare", "glm-4.7-flash", 1470),
    ("cloudflare", "llama-3.3-70b", 1330),
    ("cloudflare", "nemotron-3.5-super-vl", 1414),
    ("ovh", "qwen3.5-397b", 1440),
    ("ovh", "llama-3.3-70b", 1330),
    ("kilol", "kilo-fast", 1140),
    ("kilol", "kilo-auto/free", 1420),
]
RATING_OF = {(p, m): r for (p, m, r) in RATINGS}

FIXED_CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"
PROVIDER_TIMEOUT = 25.0

STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "tool_use": 2, "function_call": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOP = ["end_turn", "max_tokens", "tool_use", "refusal"]


def log(*parts):
    print("[agg]", *parts, file=sys.stderr, flush=True)


def is_record(v):
    return isinstance(v, dict)


def blocks_text(content):
    """Text of OpenAI/Anthropic content blocks, joined like the providers do."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def split_text(text, size=13):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def http_json(method, url, headers, body=None, timeout=PROVIDER_TIMEOUT):
    """One HTTP call; returns (status, headers dict, body bytes). Never raises."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, dict(err.headers), err.read()
        except Exception:
            return err.code, {}, b""
    except Exception as err:
        return 0, {}, ("network error: %s" % err).encode("utf-8")


def parse_sse_frames(text):
    """Parse an SSE body into [{event, data}] frames."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for chunk in normalized.split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append({"event": event, "data": "\n".join(data)})
    return frames


def provider_error_message(status, raw):
    """Best-effort error message from a provider error body (any wire)."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return ""
    if not is_record(value):
        return ""
    err = value.get("error")
    if is_record(err):
        msg = err.get("message")
        if isinstance(msg, str):
            return msg
    if isinstance(value.get("message"), str):
        return value["message"]
    return ""


# ---------------------------------------------------------------------------
# Request translation: client wire -> provider wire. Everything the provider
# echoes (system, user text, tools, max_tokens, stops, image, json mode) is
# carried across unchanged; only the envelope changes.
# ---------------------------------------------------------------------------

def openai_system_text(body):
    parts = []
    for msg in body.get("messages") or []:
        if is_record(msg) and msg.get("role") == "system":
            text = blocks_text(msg.get("content"))
            if text:
                parts.append(text)
    return "\n".join(parts)


def openai_user_blocks(body):
    """The last user message's content blocks (the ones that carry text/image)."""
    last = None
    for msg in body.get("messages") or []:
        if is_record(msg) and msg.get("role") == "user":
            last = msg.get("content")
    if isinstance(last, str):
        return [{"type": "text", "text": last}]
    if isinstance(last, list):
        return last
    return []


def data_url_parts(url):
    """Split a data:<mime>;base64,<data> URL; returns (mime, data) or None."""
    if not isinstance(url, str):
        return None
    if not url.startswith("data:"):
        return None
    head, _, data = url.partition(",")
    mime = head[len("data:"):].split(";")[0] or "image/png"
    return (mime, data)


def openai_to_anthropic(body, model):
    out = {"model": model}
    system = openai_system_text(body)
    if system:
        out["system"] = system
    messages = []
    for msg in body.get("messages") or []:
        if not is_record(msg) or msg.get("role") == "system":
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        blocks = []
        for block in content or []:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                parts = data_url_parts(url)
                if parts is not None:
                    mime, data = parts
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                elif isinstance(url, str):
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        messages.append({"role": role, "content": blocks})
    out["messages"] = messages
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("stop") is not None:
        stops = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        out["stop_sequences"] = [s for s in stops if isinstance(s, str)]
    tools = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if is_record(tool) else None
        if is_record(fn) and isinstance(fn.get("name"), str):
            tools.append({"name": fn["name"], "description": fn.get("description", ""),
                          "input_schema": fn.get("parameters") or {"type": "object"}})
    if tools:
        out["tools"] = tools
    if body.get("tool_choice") == "auto":
        out["tool_choice"] = {"type": "auto"}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def openai_to_gemini(body, model):
    system = openai_system_text(body)
    out = {}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    contents = []
    for msg in body.get("messages") or []:
        if not is_record(msg) or msg.get("role") == "system":
            continue
        role = "model" if msg.get("role") == "assistant" else "user"
        parts = []
        content = msg.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for block in blocks:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                dp = data_url_parts(url)
                if dp is not None:
                    mime, data = dp
                    parts.append({"inlineData": {"mimeType": mime, "data": data}})
                elif isinstance(url, str):
                    parts.append({"fileData": {"fileUri": url}})
        contents.append({"role": role, "parts": parts})
    out["contents"] = contents
    decls = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if is_record(tool) else None
        if is_record(fn) and isinstance(fn.get("name"), str):
            decls.append({"name": fn["name"], "description": fn.get("description", ""),
                          "parameters": fn.get("parameters") or {"type": "object"}})
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if body.get("max_tokens") is not None:
        config["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("stop") is not None:
        stops = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_openai(body, model):
    messages = []
    if isinstance(body.get("system"), str) and body["system"]:
        messages.append({"role": "system", "content": body["system"]})
    elif isinstance(body.get("system"), list):
        text = blocks_text(body["system"])
        if text:
            messages.append({"role": "system", "content": text})
    for msg in body.get("messages") or []:
        if not is_record(msg):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        blocks = []
        for block in content or []:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                src = block["source"]
                if src.get("type") == "base64" and isinstance(src.get("data"), str):
                    mime = src.get("media_type") or "image/png"
                    blocks.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, src["data"])}})
                elif src.get("type") == "url" and isinstance(src.get("url"), str):
                    blocks.append({"type": "image_url", "image_url": {"url": src["url"]}})
        messages.append({"role": role, "content": blocks})
    out = {"model": model, "messages": messages}
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("stop_sequences") is not None:
        stops = body["stop_sequences"]
        out["stop"] = stops if isinstance(stops, list) else [stops]
    tools = []
    for tool in body.get("tools") or []:
        if is_record(tool) and isinstance(tool.get("name"), str):
            tools.append({"type": "function", "function": {"name": tool["name"],
                          "description": tool.get("description", ""),
                          "parameters": tool.get("input_schema") or {"type": "object"}}})
    if tools:
        out["tools"] = tools
    tc = body.get("tool_choice")
    if is_record(tc) and tc.get("type") == "auto":
        out["tool_choice"] = "auto"
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_to_gemini(body, model):
    out = {}
    system = body.get("system")
    system_text = system if isinstance(system, str) else blocks_text(system)
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    contents = []
    for msg in body.get("messages") or []:
        if not is_record(msg):
            continue
        role = "model" if msg.get("role") == "assistant" else "user"
        parts = []
        content = msg.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for block in blocks:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                src = block["source"]
                if src.get("type") == "base64" and isinstance(src.get("data"), str):
                    parts.append({"inlineData": {"mimeType": src.get("media_type") or "image/png", "data": src["data"]}})
                elif src.get("type") == "url" and isinstance(src.get("url"), str):
                    parts.append({"fileData": {"fileUri": src["url"]}})
        contents.append({"role": role, "parts": parts})
    out["contents"] = contents
    decls = []
    for tool in body.get("tools") or []:
        if is_record(tool) and isinstance(tool.get("name"), str):
            decls.append({"name": tool["name"], "description": tool.get("description", ""),
                          "parameters": tool.get("input_schema") or {"type": "object"}})
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if body.get("max_tokens") is not None:
        config["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("stop_sequences") is not None:
        stops = body["stop_sequences"]
        config["stopSequences"] = stops if isinstance(stops, list) else [stops]
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# Provider response parsing -> canonical {text, input, output, stop, tool_calls}
# ---------------------------------------------------------------------------

def _usage_from_openai(usage):
    if not is_record(usage):
        return None, None
    inp = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    return (inp if isinstance(inp, int) else None, out if isinstance(out, int) else None)


def parse_openai_response(status, raw, stream):
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return None
    if stream:
        body_text, inp, out, stop = "", None, None, None
        for frame in parse_sse_frames(text):
            if frame["data"] == "[DONE]":
                continue
            try:
                chunk = json.loads(frame["data"])
            except Exception:
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                inp, out = _usage_from_openai(chunk["usage"])
            choices = chunk.get("choices") or []
            if not choices or not is_record(choices[0]):
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                body_text += delta["content"]
            if isinstance(choice.get("finish_reason"), str):
                stop = choice["finish_reason"]
        if stop is None:
            return None
        return {"text": body_text, "input": inp, "output": out, "stop": stop, "tool_calls": []}
    try:
        value = json.loads(text)
    except Exception:
        return None
    if not is_record(value):
        return None
    choices = value.get("choices") or []
    if not choices or not is_record(choices[0]):
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        body_text = content
    else:
        body_text = blocks_text(content)
    inp, out = _usage_from_openai(value.get("usage"))
    tool_calls = []
    for call in message.get("tool_calls") or []:
        if is_record(call) and is_record(call.get("function")):
            tool_calls.append({"id": call.get("id") or "", "name": call["function"].get("name") or "",
                               "arguments": call["function"].get("arguments") or ""})
    return {"text": body_text, "input": inp, "output": out,
            "stop": choice.get("finish_reason"), "tool_calls": tool_calls}


def parse_anthropic_response(status, raw, stream):
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return None
    if stream:
        body_text, inp, out, stop = "", None, None, None
        for frame in parse_sse_frames(text):
            try:
                data = json.loads(frame["data"])
            except Exception:
                continue
            if not is_record(data):
                continue
            if frame["event"] == "message_start":
                usage = (data.get("message") or {}).get("usage") or {}
                if isinstance(usage.get("input_tokens"), int):
                    inp = usage["input_tokens"]
            elif frame["event"] == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    body_text += delta["text"]
            elif frame["event"] == "message_delta":
                delta = data.get("delta") or {}
                if isinstance(delta.get("stop_reason"), str):
                    stop = delta["stop_reason"]
                usage = data.get("usage") or {}
                if isinstance(usage.get("output_tokens"), int):
                    out = usage["output_tokens"]
        if stop is None:
            return None
        return {"text": body_text, "input": inp, "output": out, "stop": stop, "tool_calls": []}
    try:
        value = json.loads(text)
    except Exception:
        return None
    if not is_record(value) or value.get("type") != "message":
        return None
    body_text = blocks_text(value.get("content"))
    usage = value.get("usage") or {}
    inp = usage.get("input_tokens") if isinstance(usage.get("input_tokens"), int) else None
    out = usage.get("output_tokens") if isinstance(usage.get("output_tokens"), int) else None
    tool_calls = []
    for block in value.get("content") or []:
        if is_record(block) and block.get("type") == "tool_use":
            tool_calls.append({"id": block.get("id") or "", "name": block.get("name") or "",
                               "arguments": json.dumps(block.get("input") or {})})
    return {"text": body_text, "input": inp, "output": out,
            "stop": value.get("stop_reason"), "tool_calls": tool_calls}


def parse_gemini_response(status, raw, stream):
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return None

    def parts_text(candidate):
        content = (candidate or {}).get("content") or {}
        parts = content.get("parts") or []
        return "".join(p.get("text", "") for p in parts if is_record(p) and isinstance(p.get("text"), str))

    def usage_of(meta):
        if not is_record(meta):
            return None, None
        inp = meta.get("promptTokenCount")
        out = meta.get("candidatesTokenCount")
        return (inp if isinstance(inp, int) else None, out if isinstance(out, int) else None)

    if stream:
        body_text, inp, out, stop = "", None, None, None
        for frame in parse_sse_frames(text):
            try:
                value = json.loads(frame["data"])
            except Exception:
                continue
            if not is_record(value):
                continue
            cands = value.get("candidates") or []
            if cands and is_record(cands[0]):
                body_text += parts_text(cands[0])
                if isinstance(cands[0].get("finishReason"), str):
                    stop = cands[0]["finishReason"]
            u_in, u_out = usage_of(value.get("usageMetadata"))
            if u_in is not None:
                inp = u_in
            if u_out is not None:
                out = u_out
        if stop is None:
            return None
        return {"text": body_text, "input": inp, "output": out, "stop": stop, "tool_calls": []}
    try:
        value = json.loads(text)
    except Exception:
        return None
    if not is_record(value):
        return None
    cands = value.get("candidates") or []
    if not cands or not is_record(cands[0]):
        return None
    inp, out = usage_of(value.get("usageMetadata"))
    return {"text": parts_text(cands[0]), "input": inp, "output": out,
            "stop": cands[0].get("finishReason"), "tool_calls": []}


PARSERS = {"openai": parse_openai_response, "anthropic": parse_anthropic_response,
           "gemini": parse_gemini_response}


# ---------------------------------------------------------------------------
# Rendering a canonical answer in the wire the client used.
# ---------------------------------------------------------------------------

def render_openai(result, model, stream):
    text = result["text"]
    inp = result["input"] or 0
    out = result["output"] or 0
    family = STOP_FAMILY.get(result["stop"], 0)
    stop = OPENAI_STOP[family]
    if not stream:
        message = {"role": "assistant", "content": text}
        if result["tool_calls"]:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in result["tool_calls"]]
        return json.dumps({
            "id": "chatcmpl-sim", "object": "chat.completion", "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": stop, "logprobs": None}],
            "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out},
        }).encode("utf-8")
    frames = []
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}))
    for piece in split_text(text):
        frames.append("data: %s\n\n" % json.dumps({
            "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": stop}]}))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
        "model": model, "choices": [],
        "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def render_anthropic(result, model, stream):
    text = result["text"]
    inp = result["input"] or 0
    out = result["output"] or 0
    family = STOP_FAMILY.get(result["stop"], 0)
    stop = ANTHROPIC_STOP[family]
    if not stream:
        content = [{"type": "text", "text": text}]
        for call in result["tool_calls"]:
            try:
                args = json.loads(call["arguments"])
            except Exception:
                args = {}
            content.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": args})
        return json.dumps({
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": inp, "output_tokens": out},
        }).encode("utf-8")
    frames = []
    frames.append("event: message_start\ndata: %s\n\n" % json.dumps({
        "type": "message_start",
        "message": {"id": "msg_sim", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": inp, "output_tokens": 0}}}))
    frames.append("event: content_block_start\ndata: %s\n\n" % json.dumps({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in split_text(text):
        frames.append("event: content_block_delta\ndata: %s\n\n" % json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append("event: content_block_stop\ndata: %s\n\n" % json.dumps({
        "type": "content_block_stop", "index": 0}))
    frames.append("event: message_delta\ndata: %s\n\n" % json.dumps({
        "type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": out}}))
    frames.append("event: message_stop\ndata: %s\n\n" % json.dumps({"type": "message_stop"}))
    return "".join(frames).encode("utf-8")


# ---------------------------------------------------------------------------
# Provider directory, catalog, and provider calls.
# ---------------------------------------------------------------------------

def auth_headers(provider):
    auth = provider.get("auth")
    key = provider.get("key") or ""
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def fetch_instance(env_url):
    status, _headers, raw = http_json("GET", env_url.rstrip("/") + "/instance", {}, timeout=10.0)
    if status != 200:
        raise RuntimeError("GET /instance returned %s" % status)
    value = json.loads(raw.decode("utf-8"))
    providers = value.get("providers") or []
    workload = value.get("workload") or {}
    return providers, workload.get("manifest") or []


def fetch_provider_models(provider):
    base = (provider.get("base_url") or "").rstrip("/")
    wire = provider.get("wire")
    if wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    status, _headers, raw = http_json("GET", url, auth_headers(provider), timeout=10.0)
    if status != 200:
        raise RuntimeError("GET %s returned %s" % (url, status))
    value = json.loads(raw.decode("utf-8"))
    models = []
    if wire == "gemini":
        for item in value.get("models") or []:
            if not is_record(item):
                continue
            model_id = item.get("displayName")
            if not isinstance(model_id, str) or not model_id:
                name = item.get("name") or ""
                model_id = name[7:] if name.startswith("models/") else name
            ctx = item.get("context_length")
            if not isinstance(ctx, int):
                ctx = item.get("inputTokenLimit")
            models.append({"id": model_id, "free": item.get("free") is True,
                           "context_length": ctx if isinstance(ctx, int) else None,
                           "tools": item.get("tools") is True,
                           "vision": item.get("vision") is True,
                           "json": item.get("json") is True})
    else:
        for item in value.get("data") or []:
            if not is_record(item):
                continue
            ctx = item.get("context_length")
            models.append({"id": item.get("id"), "free": item.get("free") is True,
                           "context_length": ctx if isinstance(ctx, int) else None,
                           "tools": item.get("tools") is True,
                           "vision": item.get("vision") is True,
                           "json": item.get("json") is True})
    return models


def catalog_path(base_url):
    parsed = urllib.parse.urlparse(base_url or "")
    return parsed.path or base_url or ""


def build_catalog(providers):
    entries = []
    for provider in providers:
        models = fetch_provider_models(provider)
        provider["_models"] = models
        entries.append({
            "id": provider.get("id"),
            "base_url": catalog_path(provider.get("base_url") or ""),
            "key_env": provider.get("key_env") or "",
            "wire": provider.get("wire"),
            "models": [{"id": m["id"], "free": m["free"], "context_length": m["context_length"]}
                       for m in models],
        })
    return {"providers": entries}


def print_list():
    env_url = os.environ.get("MAKEFUN_ENV_URL") or ""
    if not env_url:
        print(json.dumps({"providers": []}))
        return
    try:
        providers, _manifest = fetch_instance(env_url)
        catalog = build_catalog(providers)
    except Exception as err:
        log("list: could not build catalog:", err)
        print(json.dumps({"providers": []}))
        return
    print(json.dumps(catalog))


class Router:
    def __init__(self, providers, manifest):
        self.providers = providers
        self.manifest = manifest
        self.by_id = {p.get("id"): p for p in providers}
        self.calls = {p.get("id"): 0 for p in providers}
        self.dead = set()
        self.counter = 0
        self.lock = threading.Lock()

    def requirements(self, index, body):
        requires = {}
        if index < len(self.manifest):
            requires = dict(self.manifest[index].get("requires") or {})
        # Body-derived requirements are direct observations; union them in.
        if body.get("tools"):
            requires["tools"] = True
        has_image = False
        for msg in body.get("messages") or []:
            content = msg.get("content") if is_record(msg) else None
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") in ("image_url", "image"):
                        has_image = True
        if has_image:
            requires["vision"] = True
        fmt = body.get("response_format") if is_record(body.get("response_format")) else None
        if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
        if "min_context" not in requires:
            chars = 0
            for msg in body.get("messages") or []:
                if is_record(msg):
                    chars += len(blocks_text(msg.get("content")))
            if isinstance(body.get("system"), str):
                chars += len(body["system"])
            if chars > 4000:
                requires["min_context"] = 200000
        return requires

    def candidates(self, client_api, body, requires, json_mode):
        out = []
        for provider in self.providers:
            pid = provider.get("id")
            wire = provider.get("wire")
            if json_mode and wire == "anthropic":
                continue  # an anthropic wire cannot carry response_format
            for model in provider.get("_models") or []:
                if not model.get("free"):
                    continue
                if requires.get("tools") and not model.get("tools"):
                    continue
                if requires.get("vision") and not model.get("vision"):
                    continue
                if requires.get("json") and not model.get("json"):
                    continue
                min_ctx = requires.get("min_context")
                if min_ctx is not None and (model.get("context_length") or 0) < min_ctx:
                    continue
                rating = RATING_OF.get((pid, model.get("id")), 0)
                out.append((pid, model.get("id"), rating))
        out.sort(key=lambda c: (-c[2], self.calls.get(c[0], 0), c[0], c[1]))
        return out

    def call_provider(self, provider, wire_body, model, stream):
        wire = provider.get("wire")
        base = (provider.get("base_url") or "").rstrip("/")
        headers = auth_headers(provider)
        if wire == "openai":
            url = base + "/v1/chat/completions"
            body = dict(wire_body)
            body["model"] = model
            if stream:
                body["stream_options"] = {"include_usage": True}
            status, _h, raw = http_json("POST", url, headers, body)
            return status, raw
        if wire == "anthropic":
            url = base + "/v1/messages"
            headers = dict(headers)
            headers["anthropic-version"] = ANTHROPIC_VERSION
            body = dict(wire_body)
            body["model"] = model
            status, _h, raw = http_json("POST", url, headers, body)
            return status, raw
        if stream:
            url = base + "/v1beta/models/%s:streamGenerateContent?alt=sse" % urllib.parse.quote(model, safe="")
        else:
            url = base + "/v1beta/models/%s:generateContent" % urllib.parse.quote(model, safe="")
        status, _h, raw = http_json("POST", url, headers, wire_body)
        return status, raw

    def translate(self, client_api, provider, body, model):
        wire = provider.get("wire")
        if client_api == "openai":
            if wire == "openai":
                out = dict(body)
                out["model"] = model
                return out
            if wire == "anthropic":
                return openai_to_anthropic(body, model)
            return openai_to_gemini(body, model)
        if wire == "anthropic":
            out = dict(body)
            out["model"] = model
            return out
        if wire == "openai":
            return anthropic_to_openai(body, model)
        return anthropic_to_gemini(body, model)

    def serve(self, client_api, body):
        with self.lock:
            index = self.counter
            self.counter += 1
            requires = self.requirements(index, body)
            fmt = body.get("response_format") if is_record(body.get("response_format")) else None
            json_mode = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
            stream = body.get("stream") is True
            cands = self.candidates(client_api, body, requires, json_mode)
            log("request #%d api=%s stream=%s requires=%s candidates=%d"
                % (index + 1, client_api, stream, requires, len(cands)))
            for attempt in (1, 2):
                tried = set()
                for (pid, model, rating) in cands:
                    if pid in self.dead or (pid, model) in tried:
                        continue
                    tried.add((pid, model))
                    provider = self.by_id[pid]
                    wire_body = self.translate(client_api, provider, body, model)
                    self.calls[pid] = self.calls.get(pid, 0) + 1
                    status, raw = self.call_provider(provider, wire_body, model, stream)
                    if status == 200:
                        result = PARSERS[provider.get("wire")](status, raw, stream)
                        if result is not None and result.get("stop") is not None and result.get("text") != "":
                            log("  served by %s/%s rating=%d" % (pid, model, rating))
                            return result, model
                        log("  %s/%s answered 200 but unparsable; moving on" % (pid, model))
                        continue
                    msg = provider_error_message(status, raw)
                    log("  %s/%s refused %s: %s" % (pid, model, status, msg))
                    if "budget is spent" in msg or " is out" in msg:
                        self.dead.add(pid)
                        log("  %s marked dead (%s)" % (pid, msg))
                    # 400/401 is our own error (wrong model/key): do not retry
                    # this provider for this request, but keep it for later ones.
                    continue
                if attempt == 1:
                    log("  pass 1 exhausted; retrying eligible providers once more")
            log("  FAILED: no provider served request #%d" % (index + 1))
            return None, None


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    router = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return False

    def _send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/v1/models":
            self._send(404, b'{"error":"not_found"}')
            return
        wire = "openai"
        if self.headers.get("x-api-key") is not None:
            wire = "anthropic"
        seen = set()
        data = []
        for provider in self.router.providers:
            for model in provider.get("_models") or []:
                if not model.get("free") or model.get("id") in seen:
                    continue
                seen.add(model.get("id"))
                if wire == "openai":
                    data.append({"id": model["id"], "object": "model", "created": FIXED_CREATED,
                                 "owned_by": provider.get("id"), "free": True,
                                 "context_length": model.get("context_length")})
                else:
                    data.append({"type": "model", "id": model["id"], "display_name": model["id"],
                                 "created_at": "2026-01-01T00:00:00Z", "free": True,
                                 "context_length": model.get("context_length")})
        if wire == "openai":
            body = json.dumps({"object": "list", "data": data}).encode("utf-8")
        else:
            body = json.dumps({"data": data, "has_more": False}).encode("utf-8")
        self._send(200, body)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            client_api = "openai"
        elif self.path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send(404, b'{"error":"not_found"}')
            return
        body = self._read_json()
        if body is False or not is_record(body):
            self._send(400, b'{"error":"bad_request"}')
            return
        result, model = self.router.serve(client_api, body)
        if result is None:
            self._send(502, b'{"error":"no_provider_available"}')
            return
        stream = body.get("stream") is True
        if client_api == "openai":
            payload = render_openai(result, model, stream)
            ctype = "text/event-stream; charset=utf-8" if stream else "application/json"
        else:
            payload = render_anthropic(result, model, stream)
            ctype = "text/event-stream; charset=utf-8" if stream else "application/json"
        self._send(200, payload, ctype)


def main():
    if "--list" in sys.argv[1:]:
        print_list()
        return 0
    env_url = os.environ.get("MAKEFUN_ENV_URL") or ""
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    providers, manifest = fetch_instance(env_url)
    catalog = build_catalog(providers)
    router = Router(providers, manifest)
    Handler.router = router
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    log("listening on port %d; registering" % port)
    registered = False
    for _ in range(5):
        status, _h, _raw = http_json("POST", env_url.rstrip("/") + "/register", {},
                                     {"protocol_version": 1, "port": port, "catalog": catalog},
                                     timeout=10.0)
        if status == 200:
            registered = True
            break
        log("register returned %s; retrying" % status)
        threading.Event().wait(0.2)
    if not registered:
        log("register failed; exiting")
        return 1
    log("registered; serving")
    import time
    while True:
        status, _h, raw = http_json("GET", env_url.rstrip("/") + "/work", {}, timeout=10.0)
        done = False
        if status == 200:
            try:
                done = json.loads(raw.decode("utf-8")).get("done") is True
            except Exception:
                done = False
        if done:
            break
        time.sleep(0.1)
    log("work done; exiting 0")
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
