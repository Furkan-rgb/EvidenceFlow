"""Deterministic, section-aware Markdown policy parsing and chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SECTION_HEADING = re.compile(r"^##\s+(?P<section_id>\d+(?:\.\d+)*)\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True, slots=True)
class PolicySection:
    policy_id: str
    policy_title: str
    section_id: str
    section_title: str
    text: str
    source_path: str


@dataclass(frozen=True, slots=True)
class PolicyChunk:
    evidence_id: str
    policy_id: str
    title: str
    section_id: str
    text: str
    source_path: str


@dataclass(frozen=True, slots=True)
class PolicyCorpus:
    chunks: list[PolicyChunk]
    document_count: int
    sha256: str


def load_policy_corpus(
    policies_dir: Path,
    *,
    target_characters: int = 1000,
    max_characters: int = 1200,
    overlap_characters: int = 150,
) -> PolicyCorpus:
    """Load all policies in stable order and preserve section metadata."""

    _validate_chunk_settings(target_characters, max_characters, overlap_characters)
    paths = sorted(policies_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown policies found in {policies_dir}")

    digest = hashlib.sha256()
    chunks: list[PolicyChunk] = []
    seen_policy_ids: set[str] = set()
    seen_section_ids: set[tuple[str, str]] = set()
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(policies_dir).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw.encode("utf-8"))
        digest.update(b"\0")

        policy_id, title, markdown = _parse_front_matter(path, raw)
        if policy_id in seen_policy_ids:
            raise ValueError(f"Duplicate policy_id: {policy_id}")
        seen_policy_ids.add(policy_id)
        sections = _parse_sections(policy_id, title, relative_path, markdown)
        for section in sections:
            section_key = (section.policy_id, section.section_id)
            if section_key in seen_section_ids:
                raise ValueError(
                    f"Duplicate section_id {section.section_id} in {section.policy_id}"
                )
            seen_section_ids.add(section_key)
            section_chunks = split_text(
                section.text,
                target_characters=target_characters,
                max_characters=max_characters,
                overlap_characters=overlap_characters,
            )
            for chunk_index, text in enumerate(section_chunks):
                chunks.append(
                    PolicyChunk(
                        evidence_id=(
                            f"{section.policy_id}:{section.section_id}:chunk-{chunk_index}"
                        ),
                        policy_id=section.policy_id,
                        title=section.policy_title,
                        section_id=section.section_id,
                        text=f"{section.section_title}\n\n{text}",
                        source_path=section.source_path,
                    )
                )

    return PolicyCorpus(
        chunks=chunks,
        document_count=len(paths),
        sha256=digest.hexdigest(),
    )


def split_text(
    text: str,
    *,
    target_characters: int = 1000,
    max_characters: int = 1200,
    overlap_characters: int = 150,
) -> list[str]:
    """Split on readable boundaries with deterministic character overlap."""

    _validate_chunk_settings(target_characters, max_characters, overlap_characters)
    normalized = re.sub(r"[ \t]+", " ", text.strip())
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if not normalized:
        return []
    if len(normalized) <= max_characters:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        remaining = len(normalized) - start
        if remaining <= max_characters:
            end = len(normalized)
        else:
            desired_end = min(start + target_characters, len(normalized))
            hard_end = min(start + max_characters, len(normalized))
            end = _best_boundary(normalized, start, desired_end, hard_end)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap_characters)
        while next_start < end and not normalized[next_start - 1].isspace():
            next_start += 1
        start = min(next_start, end)
    return chunks


def _best_boundary(text: str, start: int, desired_end: int, hard_end: int) -> int:
    for separator in ("\n\n", ". ", "\n", " "):
        position = text.rfind(separator, start + 1, hard_end + 1)
        if position >= desired_end - 200:
            return position + len(separator.rstrip())
    return hard_end


def _parse_front_matter(path: Path, raw: str) -> tuple[str, str, str]:
    if not raw.startswith("---\n"):
        raise ValueError(f"Policy {path} is missing YAML front matter")
    closing = raw.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"Policy {path} has unterminated YAML front matter")
    metadata = yaml.safe_load(raw[4:closing])
    if not isinstance(metadata, dict):
        raise ValueError(f"Policy {path} front matter must be a mapping")
    policy_id = metadata.get("policy_id")
    title = metadata.get("title")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"Policy {path} must define policy_id")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Policy {path} must define title")
    return policy_id.strip(), title.strip(), raw[closing + 5 :]


def _parse_sections(
    policy_id: str, policy_title: str, source_path: str, markdown: str
) -> list[PolicySection]:
    sections: list[PolicySection] = []
    current_id: str | None = None
    current_title: str | None = None
    current_lines: list[str] = []

    def finish_section() -> None:
        if current_id is None or current_title is None:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            raise ValueError(f"Empty policy section {policy_id} §{current_id}")
        sections.append(
            PolicySection(
                policy_id=policy_id,
                policy_title=policy_title,
                section_id=current_id,
                section_title=current_title,
                text=text,
                source_path=source_path,
            )
        )

    for line in markdown.splitlines():
        match = _SECTION_HEADING.match(line)
        if match:
            finish_section()
            current_id = match.group("section_id")
            current_title = match.group("title").strip()
            current_lines = []
        elif current_id is not None:
            current_lines.append(line)
    finish_section()
    if not sections:
        raise ValueError(f"Policy {source_path} has no stable level-two sections")
    return sections


def _validate_chunk_settings(target: int, maximum: int, overlap: int) -> None:
    if target < 1 or maximum < target:
        raise ValueError("chunk target must be positive and no greater than the maximum")
    if overlap < 0 or overlap >= target:
        raise ValueError("chunk overlap must be non-negative and smaller than target")
