"""
emoji_script.py — alias lookup index and renderer for the Emoji News Script Ontology.

This module is the middle layer of the news-to-emoji translator pipeline:

    ontology JSON  ->  AliasIndex (text -> concept refs)
                   ->  StoryRenderer (structured story -> emoji string)

A "structured story" is the intermediate representation the LLM will be asked
to produce in the next stage. It is a JSON document of token lists, where each
token is one of:

    {"ref": "health_minister"}                  # ontology entry (any section)
    {"ref": "school", "plural": true}           # plural marker appended
    {"grammar": "past_tense"}                   # grammar marker by key
    {"name": "Wes Streeting", "kind": "person"} # wrapped proper noun
                                                #   kind: person|place|org|generic
    {"number": 250}                             # numeral pass-through
    {"text": "..."}                             # untranslatable literal (escape hatch)

Validation is deliberately strict: every ref and grammar key must exist in the
ontology, so malformed LLM output fails fast rather than rendering garbage.

CLI:
    python emoji_script.py validate                      # check ontology integrity
    python emoji_script.py match "Health secretary announces NHS funding"
    python emoji_script.py render story.json
    python emoji_script.py demo                          # end-to-end smoke test
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ONTOLOGY_PATH = Path(__file__).parent / "emoji_news_ontology.json"

# Sections containing alias-bearing entries, in priority order for conflict
# resolution: when the same alias appears in two sections, the earlier section
# wins. Compounds outrank concepts because they are more specific.
ALIAS_SECTIONS = ("compounds", "concepts", "actions", "modifiers")

WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

NAME_WRAPPERS = {
    "person": "person_name_wrapper",
    "place": "place_name_wrapper",
    "org": "organisation_name_wrapper",
    "generic": "proper_name_wrapper",
}


# --------------------------------------------------------------------------
# Ontology loading and integrity checks
# --------------------------------------------------------------------------

class OntologyError(Exception):
    """Raised when the ontology file is structurally unsound."""


def load_ontology(path: str | Path = ONTOLOGY_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        ont = json.load(fh)
    errors = validate_ontology(ont)
    if errors:
        raise OntologyError(
            "Ontology failed validation:\n  " + "\n  ".join(errors)
        )
    return ont


def validate_ontology(ont: dict) -> list[str]:
    """Return a list of integrity errors (empty list == valid)."""
    errors: list[str] = []

    for section in ("grammar", "concepts", "compounds"):
        if section not in ont:
            errors.append(f"missing required section: {section}")
    if errors:
        return errors

    known_refs = set()
    for section in ("concepts", "actions", "modifiers"):
        for key, entry in ont.get(section, {}).items():
            if key in known_refs:
                errors.append(f"duplicate entry key across sections: {key}")
            known_refs.add(key)
            if not entry.get("emoji"):
                errors.append(f"{section}.{key}: missing emoji glyph")

    for key, comp in ont.get("compounds", {}).items():
        seq = comp.get("sequence", [])
        if not seq:
            errors.append(f"compounds.{key}: empty sequence")
        for ref in seq:
            if ref not in known_refs:
                errors.append(f"compounds.{key}: unresolved sequence ref '{ref}'")
        if key in known_refs:
            errors.append(f"compounds.{key}: key collides with a concept/action/modifier")

    return errors


# --------------------------------------------------------------------------
# Alias index — longest-match-first lookup
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Match:
    start: int          # word offset in the tokenised input
    end: int            # exclusive word offset
    surface: str        # the text that matched
    ref: str            # ontology entry key
    section: str        # which section the entry lives in
    emoji: str          # rendered glyph(s) for quick inspection


@dataclass
class AliasIndex:
    """
    Maps lowercase alias word-tuples to ontology refs and scans text using
    greedy longest-match-first resolution, so 'health secretary' resolves to
    the health_minister compound before 'health' and 'secretary' can match
    individually.
    """
    ontology: dict
    _index: dict[tuple[str, ...], tuple[str, str]] = field(default_factory=dict)
    _max_len: int = 1
    conflicts: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, ontology: dict) -> "AliasIndex":
        idx = cls(ontology=ontology)
        for section in ALIAS_SECTIONS:
            for key, entry in ontology.get(section, {}).items():
                aliases = set(a.lower() for a in entry.get("aliases", []))
                # The entry key itself is matchable too (underscores -> spaces).
                aliases.add(key.replace("_", " "))
                for alias in aliases:
                    words = tuple(WORD_RE.findall(alias))
                    if not words:
                        continue
                    if words in idx._index and idx._index[words][1] != key:
                        prior_section, prior_key = idx._index[words]
                        idx.conflicts.append(
                            f"alias '{alias}' -> {section}.{key} ignored; "
                            f"already bound to {prior_section}.{prior_key}"
                        )
                        continue  # first binding wins (section priority order)
                    idx._index[words] = (section, key)
                    idx._max_len = max(idx._max_len, len(words))
        return idx

    def lookup(self, phrase: str) -> tuple[str, str] | None:
        """Exact lookup of a phrase. Returns (section, ref) or None."""
        return self._index.get(tuple(WORD_RE.findall(phrase.lower())))

    def scan(self, text: str) -> list[Match]:
        """Greedy longest-match scan across free text."""
        words = WORD_RE.findall(text.lower())
        matches: list[Match] = []
        i = 0
        while i < len(words):
            hit = None
            for span in range(min(self._max_len, len(words) - i), 0, -1):
                candidate = tuple(words[i : i + span])
                if candidate in self._index:
                    section, ref = self._index[candidate]
                    hit = Match(
                        start=i,
                        end=i + span,
                        surface=" ".join(candidate),
                        ref=ref,
                        section=section,
                        emoji=entry_glyph(self.ontology, section, ref),
                    )
                    break
            if hit:
                matches.append(hit)
                i = hit.end
            else:
                i += 1
        return matches


def entry_glyph(ont: dict, section: str, ref: str) -> str:
    entry = ont[section][ref]
    if section == "compounds":
        return entry.get("render") or render_compound(ont, ref)
    return entry["emoji"]


# --------------------------------------------------------------------------
# Renderer — structured story -> emoji string
# --------------------------------------------------------------------------

class StoryValidationError(Exception):
    """Raised when a structured story references unknown ontology entries."""


def render_compound(ont: dict, key: str) -> str:
    """Build a compound's glyph string from its sequence (fallback if no render)."""
    g = ont["grammar"]
    comp = ont["compounds"][key]
    inner = ""
    for ref in comp["sequence"]:
        for section in ("concepts", "actions", "modifiers"):
            if ref in ont.get(section, {}):
                inner += ont[section][ref]["emoji"]
                break
    return g["compound_open"] + inner + g["compound_close"]


@dataclass
class StoryRenderer:
    ontology: dict

    def _resolve_ref(self, ref: str) -> str:
        for section in ALIAS_SECTIONS:
            if ref in self.ontology.get(section, {}):
                return entry_glyph(self.ontology, section, ref)
        raise StoryValidationError(f"unknown ref: '{ref}'")

    def render_token(self, token: dict) -> str:
        g = self.ontology["grammar"]

        if "ref" in token:
            glyph = self._resolve_ref(token["ref"])
            if token.get("plural"):
                glyph += g["plural"]
            return glyph

        if "grammar" in token:
            key = token["grammar"]
            if key not in g:
                raise StoryValidationError(f"unknown grammar marker: '{key}'")
            return g[key]

        if "name" in token:
            kind = token.get("kind", "generic")
            wrapper_key = NAME_WRAPPERS.get(kind)
            if wrapper_key is None:
                raise StoryValidationError(f"unknown name kind: '{kind}'")
            w = g[wrapper_key]
            return f"{w}{token['name']}{w}"

        if "number" in token:
            return f"{g['quantity_marker']}{token['number']}"

        if "text" in token:
            return token["text"]

        raise StoryValidationError(f"unrecognised token shape: {token}")

    def render_tokens(self, tokens: list[dict]) -> str:
        return " ".join(self.render_token(t) for t in tokens)

    def validate_story(self, story: dict) -> list[str]:
        """Dry-run every token; return a list of error strings."""
        errors = []
        for where, tokens in self._iter_token_lists(story):
            for n, token in enumerate(tokens):
                try:
                    self.render_token(token)
                except StoryValidationError as exc:
                    errors.append(f"{where}[{n}]: {exc}")
        return errors

    def render_story(self, story: dict) -> str:
        errors = self.validate_story(story)
        if errors:
            raise StoryValidationError(
                "story failed validation:\n  " + "\n  ".join(errors)
            )
        lines = []
        if "headline" in story:
            g = self.ontology["grammar"]
            lines.append(g["breaking_news"] + " " + self.render_tokens(story["headline"]))
        for sentence in story.get("sentences", []):
            lines.append(self.render_tokens(sentence))
        return "\n".join(lines)

    @staticmethod
    def _iter_token_lists(story: dict) -> Iterator[tuple[str, list[dict]]]:
        if "headline" in story:
            yield "headline", story["headline"]
        for i, sentence in enumerate(story.get("sentences", [])):
            yield f"sentences[{i}]", sentence


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DEMO_STORY = {
    "headline": [
        {"ref": "health_minister"},
        {"grammar": "past_tense"},
        {"ref": "announce"},
        {"grammar": "new"},
        {"ref": "fund"},
        {"ref": "nhs"},
    ],
    "sentences": [
        [
            {"name": "Wes Streeting", "kind": "person"},
            {"ref": "say"},
            {"grammar": "quote_open"},
            {"ref": "money"},
            {"grammar": "direction"},
            {"ref": "hospital_waiting_lists"},
            {"grammar": "decrease"},
            {"grammar": "quote_close"},
        ],
        [
            {"ref": "union"},
            {"ref": "warn"},
            {"grammar": "uncertainty"},
            {"ref": "carers", "plural": True},
            {"ref": "money"},
            {"grammar": "comparison_less"},
        ],
    ],
}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd, *rest = argv
    ont = load_ontology()
    index = AliasIndex.build(ont)
    renderer = StoryRenderer(ont)

    if cmd == "validate":
        print(f"ontology OK: {len(index._index)} aliases indexed, "
              f"max phrase length {index._max_len} words")
        for warning in index.conflicts:
            print(f"  warning: {warning}")
        return 0

    if cmd == "match":
        text = " ".join(rest)
        for m in index.scan(text):
            print(f"{m.surface!r:35s} -> {m.section}.{m.ref:30s} {m.emoji}")
        return 0

    if cmd == "render":
        story = json.loads(Path(rest[0]).read_text(encoding="utf-8"))
        print(renderer.render_story(story))
        return 0

    if cmd == "demo":
        print("-- alias scan --")
        for m in index.scan(
            "The health secretary announced new NHS funding as care workers "
            "warned the CQC about home care cuts."
        ):
            print(f"  {m.surface!r:25s} -> {m.section}.{m.ref:25s} {m.emoji}")
        print("-- rendered story --")
        print(renderer.render_story(DEMO_STORY))
        return 0

    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
