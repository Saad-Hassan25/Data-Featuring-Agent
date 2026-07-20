"""Tiny helpers for loading and filling the markdown prompt templates.

Templates use `<<TOKEN>>` placeholders (not `str.format` braces) so the JSON
examples inside them never collide with substitution."""

from __future__ import annotations

from importlib import resources


def template(name: str) -> str:
    return resources.files("feature_agent.prompts").joinpath(name).read_text(encoding="utf-8")


def fill(name: str, subs: dict[str, str]) -> str:
    text = template(name)
    for k, v in subs.items():
        text = text.replace(k, v)
    return text
