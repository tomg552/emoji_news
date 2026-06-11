"""
translator.py — LLM driver for the Emoji News Script translator.

Pipeline:
    article text -> prompt (with full ontology) -> LLM -> structured story JSON
                 -> validate against ontology -> retry with errors if invalid
                 -> render emoji

The LLM client targets any OpenAI-compatible chat completions endpoint, which
covers vLLM, Hugging Face TGI (>= 1.4 exposes /v1/chat/completions), HF
Inference Endpoints, and the HF router. Configure with base_url + model +
api_key. No SDK dependency — plain requests.

Kept free of Streamlit imports so it can be unit-tested and later lifted onto
COB as a capability without UI baggage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

import requests

from emoji_script import (
    AliasIndex,
    StoryRenderer,
    load_ontology,
)

DEFAULT_TIMEOUT = 120
MAX_RETRIES = 2  # retries after the first attempt, driven by validation errors


# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------

class ChatClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


@dataclass
class OpenAICompatibleClient:
    """
    Minimal chat-completions client for vLLM / TGI / HF endpoints.

    schema_style controls constrained decoding:
      None    — plain generation; rely on the validate-and-retry loop
      "vllm"  — OpenAI-standard response_format json_schema (vLLM, and any
                provider implementing OpenAI structured outputs)
      "tgi"   — TGI's response_format {"type": "json_object", "value": schema}
    """
    base_url: str                 # e.g. https://my-endpoint.run.app/v1
    model: str                    # served model name (TGI often accepts "tgi")
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: int = DEFAULT_TIMEOUT
    schema_style: str | None = None
    json_schema: dict | None = None

    def complete(self, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_schema and self.schema_style == "vllm":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "emoji_story", "schema": self.json_schema},
            }
        elif self.json_schema and self.schema_style == "tgi":
            payload["response_format"] = {
                "type": "json_object",
                "value": self.json_schema,
            }
        resp = requests.post(
            self.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a news-to-emoji translator. You convert a news article into a
STRUCTURED STORY: a JSON object that references entries in the emoji
ontology supplied below. You never invent ontology keys.

OUTPUT FORMAT — respond with ONLY a JSON object, no prose, no markdown fences:
{{
  "headline": [ <token>, ... ],
  "sentences": [ [ <token>, ... ], ... ]
}}

Each <token> is exactly one of:
  {{"ref": "<key>"}}                       — a key from concepts, actions,
                                             modifiers, or compounds
  {{"ref": "<key>", "plural": true}}       — plural form of a concept
  {{"grammar": "<key>"}}                   — a key from the grammar section
                                             (tense, negation, cause, etc.)
  {{"name": "<text>", "kind": "<kind>"}}   — proper noun not in the ontology;
                                             kind is person, place, org,
                                             or generic
  {{"number": <int or float>}}             — a numeral from the article
  {{"text": "<word>"}}                     — LAST RESORT for an essential word
                                             with no ontology entry; use rarely

TRANSLATION RULES:
1. Prefer compounds over sequences of single concepts when one exists
   (e.g. ref health_minister, not government + minister + health).
2. Mark tense with a grammar token BEFORE the action: past_tense for
   reported events, future_tense for planned ones.
3. Use cause (🔗) and result (⤵️) grammar tokens to link events.
4. Wrap people, places, and organisations not in the ontology as name
   tokens with the correct kind.
5. Compress: one token sentence per article sentence or key fact.
   Aim for 3 to 8 tokens per sentence. Drop filler.
6. The headline should be 3 to 6 tokens capturing the core event.
7. Every "ref" and "grammar" value MUST exist in the ontology below.
   Output is machine-validated; unknown keys are rejected.

EXAMPLE
Article: "The health secretary announced new funding for the NHS."
Output:
{example_story}

ONTOLOGY (the only keys you may reference):
{ontology_json}
"""

RETRY_TEMPLATE = """\
Your previous output failed validation with these errors:
{errors}

Fix ONLY the offending tokens. Every "ref" must be a key from concepts,
actions, modifiers, or compounds; every "grammar" must be a key from the
grammar section. If no suitable key exists, use a name token or a text
token instead. Respond with ONLY the corrected JSON object.
"""

EXAMPLE_STORY = {
    "headline": [
        {"ref": "health_minister"},
        {"grammar": "past_tense"},
        {"ref": "announce"},
        {"grammar": "new"},
        {"ref": "fund"},
        {"ref": "nhs"},
    ],
    "sentences": [],
}


def build_system_prompt(ontology: dict) -> str:
    return SYSTEM_TEMPLATE.format(
        example_story=json.dumps(EXAMPLE_STORY, ensure_ascii=False),
        ontology_json=json.dumps(ontology, ensure_ascii=False),
    )


# --------------------------------------------------------------------------
# JSON extraction — models love fences and preambles
# --------------------------------------------------------------------------

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model response, fences or not."""
    fenced = FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1)
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("model output parsed but is not a JSON object")
    return obj


# --------------------------------------------------------------------------
# Story JSON Schema — generated from the ontology so ref/grammar values are
# enums of real keys. With constrained decoding the model cannot emit an
# unknown key; the validate-and-retry loop then becomes a backstop.
# --------------------------------------------------------------------------

def build_story_schema(ontology: dict) -> dict:
    refs = sorted(
        set(ontology.get("concepts", {}))
        | set(ontology.get("actions", {}))
        | set(ontology.get("modifiers", {}))
        | set(ontology.get("compounds", {}))
    )
    grammar_keys = sorted(ontology.get("grammar", {}))

    token = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "enum": refs},
                    "plural": {"type": "boolean"},
                },
                "required": ["ref"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "grammar": {"type": "string", "enum": grammar_keys},
                },
                "required": ["grammar"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["person", "place", "org", "generic"],
                    },
                },
                "required": ["name", "kind"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"number": {"type": "number"}},
                "required": ["number"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ]
    }

    return {
        "type": "object",
        "properties": {
            "headline": {"type": "array", "items": token, "minItems": 1, "maxItems": 8},
            "sentences": {
                "type": "array",
                "items": {"type": "array", "items": token, "minItems": 1, "maxItems": 14},
            },
        },
        "required": ["headline", "sentences"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# Translation loop
# --------------------------------------------------------------------------

@dataclass
class TranslationResult:
    ok: bool
    emoji: str = ""
    story: dict | None = None
    attempts: int = 0
    log: list[str] = field(default_factory=list)
    grounding: list[dict] = field(default_factory=list)  # alias-scan matches


@dataclass
class Translator:
    client: ChatClient
    ontology: dict = field(default_factory=load_ontology)
    max_retries: int = MAX_RETRIES
    on_event: Callable[[str], None] | None = None  # progress hook for the UI

    def __post_init__(self):
        self.renderer = StoryRenderer(self.ontology)
        self.index = AliasIndex.build(self.ontology)
        self.system_prompt = build_system_prompt(self.ontology)

    def _emit(self, msg: str, log: list[str]) -> None:
        log.append(msg)
        if self.on_event:
            self.on_event(msg)

    def translate(self, article: str) -> TranslationResult:
        result = TranslationResult(ok=False)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Translate this article:\n\n{article}"},
        ]

        # Deterministic grounding pass — what the alias index finds in the
        # source text. Shown alongside the LLM output for sanity checking.
        result.grounding = [
            {"surface": m.surface, "ref": m.ref, "section": m.section, "emoji": m.emoji}
            for m in self.index.scan(article)
        ]

        for attempt in range(1, self.max_retries + 2):
            result.attempts = attempt
            self._emit(f"attempt {attempt}: calling model", result.log)
            raw = self.client.complete(messages)

            try:
                story = extract_json(raw)
            except ValueError as exc:
                self._emit(f"attempt {attempt}: JSON extraction failed: {exc}", result.log)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "That was not parseable JSON. Respond with ONLY "
                               "the JSON object, no prose, no fences.",
                })
                continue

            errors = self.renderer.validate_story(story)
            if not errors:
                result.ok = True
                result.story = story
                result.emoji = self.renderer.render_story(story)
                self._emit(f"attempt {attempt}: validated and rendered", result.log)
                return result

            self._emit(
                f"attempt {attempt}: {len(errors)} validation error(s): "
                + "; ".join(errors[:5]),
                result.log,
            )
            result.story = story  # keep last attempt for inspection
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": RETRY_TEMPLATE.format(errors="\n".join(f"- {e}" for e in errors)),
            })

        self._emit("exhausted retries without a valid story", result.log)
        return result
