"""Versioned reading-state repository for the Lue eBook reader.

A book is identified by a *stable identity* composed of whatever durable
metadata is available:

* canonical (real, absolute) file path,
* full content hash (file content summary),
* the EPUB unique identifier (``<dc:identifier>``) when present.

Every saved state additionally records the normalized text of the current
sentence together with normalized context before/after it, the source chapter
anchor (page anchor where applicable) and an update timestamp.

Writes are crash-safe and concurrency-safe: the repository is serialized to a
temporary file in the same directory, flushed/fsynced and atomically replaced
via ``os.replace``; an advisory instance lock plus monotonic revision/timestamp
comparison ensure an older write can never silently overwrite a committed
newer version.

Legacy per-book files named after the book title (``<Title>.progress.json``)
are migrated once, automatically, on first access.  The originals are kept
and additionally copied to a backup directory, so the migration is fully
rollback-able.  The recent-books menu reads exclusively from this repository.
"""

from __future__ import annotations

import os
import re
import json
import time
import shutil
import hashlib
import tempfile
import logging
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from rich.console import Console

from . import config, content_parser

try:  # POSIX instance locks; Windows falls back to revision comparison only.
    import fcntl
except ImportError:  # pragma: no cover - platform specific
    fcntl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

STATE_FILE_NAME = "state.json"
LOCK_FILE_NAME = "state.lock"
LEGACY_GLOB_PATTERN = "*.progress.json"
BACKUP_DIR_NAME = "migration_backup"

STATE_FORMAT = "lue-state"
SCHEMA_VERSION = 1

RECENT_BOOKS_MAX = 50
CONTEXT_RADIUS = 2
ANSWER_CONTEXT_LIMIT = 300
NORM_SNIPPET_LIMIT = 160
PARAGRAPH_NORM_LIMIT = 400
# Temp files older than this are safe to clean even if the owning PID exists.
STALE_TEMP_MAX_AGE = 3600.0

# Fingerprint similarity at which a moved/edited file is accepted as the same
# book even when neither path, content hash nor EPUB identifier match.
FINGERPRINT_ACCEPT = 0.50
# Fuzzy sentence matching thresholds.
FUZZY_RATIO_MIN = 0.60
FUZZY_ACCEPT_SCORE = 0.72
# Cheap pre-filter window used before running SequenceMatcher.
LENGTH_RATIO_WINDOW = 3.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization and hashing
# ---------------------------------------------------------------------------

def normalize_text(text: str | None) -> str:
    """NFKC + casefold, keeping alphanumeric characters only.

    Punctuation and whitespace are dropped so that small formatting, quote or
    spacing changes do not break anchor matching.  CJK characters are
    alphanumeric as far as :meth:`str.isalnum` is concerned, so they survive.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file_content(file_path: str, block_size: int = 1024 * 1024) -> str:
    """Stream the file and return its SHA-256 hex digest."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Stable book identity
# ---------------------------------------------------------------------------

def read_epub_identifier(file_path: str) -> str | None:
    """Return the EPUB's unique identifier from its OPF metadata, if any."""
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            try:
                container_root = ET.fromstring(archive.read("META-INF/container.xml"))
            except KeyError:
                return None
            rootfile = container_root.find(".//{*}rootfile")
            if rootfile is None:
                return None
            opf_path = rootfile.get("full-path")
            if not opf_path:
                return None
            opf_root = ET.fromstring(archive.read(opf_path.replace("\\", "/")))
        # Namespace agnostic: dc:identifier tags end with "}identifier".
        for elem in opf_root.iter():
            tag = elem.tag.lower()
            if tag == "identifier" or tag.endswith("}identifier"):
                value = (elem.text or "").strip()
                if value:
                    return value
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError):
        return None
    return None


def compute_identity(file_path: str) -> dict[str, Any]:
    """Build the stable identity descriptor for an eBook file."""
    canonical_path = os.path.realpath(file_path)
    stat_result = os.stat(canonical_path)
    content_hash = hash_file_content(canonical_path)
    ext = os.path.splitext(canonical_path)[1].lower()
    epub_identifier = read_epub_identifier(canonical_path) if ext == ".epub" else None

    # The durable seed: EPUB identifier takes precedence, content hash is
    # always mixed in so unrelated books can never collide.
    seed = "\x1f".join(["lue-book-v1", epub_identifier or "", content_hash])
    book_id = "b_" + _sha256_hex(seed.encode("utf-8"))[:32]

    return {
        "book_id": book_id,
        "canonical_path": canonical_path,
        "content_hash": content_hash,
        "epub_identifier": epub_identifier,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "title_hint": os.path.splitext(os.path.basename(canonical_path))[0],
        "known_content_hashes": [content_hash],
        "known_paths": [canonical_path],
    }


def build_fingerprint(chapters: list[list[str]]) -> dict[str, Any]:
    """Content fingerprint robust to small edits.

    Hashes of every non-empty normalized paragraph, in order, plus the hash
    of the concatenated text.  Multiset overlap of paragraph hashes measures
    whether two documents are the same book after edits.
    """
    paragraph_hashes: list[str] = []
    parts: list[str] = []
    for chapter in chapters:
        for paragraph in chapter:
            normalized = normalize_text(paragraph)
            if not normalized:
                continue
            parts.append(normalized)
            paragraph_hashes.append(_sha256_hex(normalized.encode("utf-8"))[:16])
    return {
        "text_hash": _sha256_hex("".join(parts).encode("utf-8")),
        "paragraph_hashes": paragraph_hashes,
        "paragraph_count": len(paragraph_hashes),
    }


def _multiset_jaccard(left: list[str], right: list[str]) -> float:
    def counts(values: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return result

    left_counts, right_counts = counts(left), counts(right)
    intersection = 0
    for key, count in left_counts.items():
        intersection += min(count, right_counts.get(key, 0))
    union = 0
    keys = set(left_counts) | set(right_counts)
    for key in keys:
        union += max(left_counts.get(key, 0), right_counts.get(key, 0))
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Position anchors
# ---------------------------------------------------------------------------

def flatten_sentences(chapters: list[list[str]]) -> list[tuple[int, int, int, str]]:
    """Return every sentence in reading order as (c, p, s, text)."""
    flat: list[tuple[int, int, int, str]] = []
    for c, chapter in enumerate(chapters):
        for p, paragraph in enumerate(chapter):
            for s, sentence in enumerate(content_parser.split_into_sentences(paragraph)):
                flat.append((c, p, s, sentence))
    return flat


def _chapter_anchor_name(chapters: list[list[str]], chapter_idx: int) -> str:
    if not 0 <= chapter_idx < len(chapters):
        return f"Chapter {chapter_idx + 1}"
    for paragraph in chapters[chapter_idx]:
        stripped = paragraph.strip()
        if stripped and len(stripped) > 3:
            return stripped[:80]
    return f"Chapter {chapter_idx + 1}"


def build_anchor(chapters: list[list[str]], c: int, p: int, s: int,
                 context_radius: int = CONTEXT_RADIUS) -> dict[str, Any] | None:
    """Build the normalized anchor/context payload for a reading position."""
    try:
        paragraph = chapters[c][p]
    except IndexError:
        return None
    sentences = content_parser.split_into_sentences(paragraph)
    if not sentences:
        return None
    if not 0 <= s < len(sentences):
        s = max(0, min(s, len(sentences) - 1))
    sentence = sentences[s]

    flat = flatten_sentences(chapters)
    ordinal = None
    for idx, (fc, fp, fs, _) in enumerate(flat):
        if fc == c and fp == p and fs == s:
            ordinal = idx
            break
    if ordinal is None:
        ordinal = 0

    before = [
        normalize_text(flat[k][3])[:NORM_SNIPPET_LIMIT]
        for k in range(max(0, ordinal - context_radius), ordinal)
    ]
    after = [
        normalize_text(flat[k][3])[:NORM_SNIPPET_LIMIT]
        for k in range(ordinal + 1, min(len(flat), ordinal + 1 + context_radius))
    ]

    return {
        "c": c,
        "p": p,
        "s": s,
        "ordinal": ordinal,
        "total_sentences": len(flat),
        "chapter_anchor": _chapter_anchor_name(chapters, c),
        "page_anchor": None,
        "anchor_text": sentence.strip()[:ANSWER_CONTEXT_LIMIT],
        "anchor_norm": normalize_text(sentence)[:NORM_SNIPPET_LIMIT],
        "paragraph_norm": normalize_text(paragraph)[:PARAGRAPH_NORM_LIMIT],
        "context_before": before,
        "context_after": after,
    }


def coordinates_valid(chapters: list[list[str]], c: int, p: int, s: int) -> bool:
    try:
        sentences = content_parser.split_into_sentences(chapters[c][p])
    except IndexError:
        return False
    return 0 <= s < len(sentences)


def clamp_coordinates(chapters: list[list[str]], c: int, p: int, s: int
                      ) -> tuple[int, int, int]:
    """Clamp out-of-range coordinates into the document structure."""
    if not chapters:
        return 0, 0, 0
    c = max(0, min(c, len(chapters) - 1))
    if not chapters[c]:
        return c, 0, 0
    p = max(0, min(p, len(chapters[c]) - 1))
    sentence_count = len(content_parser.split_into_sentences(chapters[c][p]))
    s = max(0, min(s, max(0, sentence_count - 1)))
    return c, p, s


def relocate_in_document(chapters: list[list[str]], stored_position: dict[str, Any]
                         ) -> dict[str, Any] | None:
    """Locate a stored anchor inside a (possibly edited) document.

    Returns ``{"c", "p", "s", "method", "score"}`` when a reliable match was
    found, otherwise ``None``.  Paragraphs inserted before the saved position
    are handled naturally because the search is content based.
    """
    anchor_norm = stored_position.get("anchor_norm") or ""
    paragraph_norm = stored_position.get("paragraph_norm") or ""
    if not anchor_norm:
        return None

    flat = flatten_sentences(chapters)
    normalized_flat = [normalize_text(item[3]) for item in flat]

    before_norms = stored_position.get("context_before", []) or []
    after_norms = stored_position.get("context_after", []) or []

    def context_score(candidate_idx: int) -> float:
        total = len(before_norms) + len(after_norms)
        if total == 0:
            return 0.0
        window = max(10, total * 5)
        matched = 0

        def nearby(target: str, direction: int) -> bool:
            if not target:
                return False
            start = candidate_idx + direction
            stop = candidate_idx + direction * (window + 1)
            indices = range(start, stop, direction)
            for k in indices:
                if k < 0 or k >= len(normalized_flat):
                    break
                candidate = normalized_flat[k]
                if not candidate:
                    continue
                if candidate == target:
                    return True
                # Containment handles slightly changed neighboring sentences.
                if len(target) >= 24 and (target in candidate or candidate in target):
                    return True
            return False

        for target in before_norms:
            if nearby(target, -1):
                matched += 1
        for target in after_norms:
            if nearby(target, 1):
                matched += 1
        return matched / total

    # ---- 1. Exact (normalized) sentence match -----------------------------
    exact_matches = [i for i, n in enumerate(normalized_flat) if n == anchor_norm]
    if not exact_matches and len(anchor_norm) >= 24:
        # Substring match tolerates minor edits inside the sentence.
        exact_matches = [
            i for i, n in enumerate(normalized_flat)
            if n and (anchor_norm in n or n in anchor_norm)
        ]
    if exact_matches:
        best_idx, best_score = exact_matches[0], -1.0
        for idx in exact_matches:
            score = context_score(idx)
            if score > best_score:
                best_idx, best_score = idx, score
        c, p, s, _ = flat[best_idx]
        return {"c": c, "p": p, "s": s, "method": "exact_anchor",
                "score": 1.0 if normalized_flat[best_idx] == anchor_norm else 0.9}

    # ---- 2. Exact paragraph match; recover sentence by relative index ----
    if paragraph_norm:
        for ci, chapter in enumerate(chapters):
            for pi, paragraph in enumerate(chapter):
                normalized_paragraph = normalize_text(paragraph)
                if normalized_paragraph and (
                    normalized_paragraph == paragraph_norm
                    or (len(paragraph_norm) >= 40
                        and (paragraph_norm in normalized_paragraph
                             or normalized_paragraph in paragraph_norm))
                ):
                    sentence_count = len(
                        content_parser.split_into_sentences(paragraph)
                    )
                    relative_s = stored_position.get("s", 0)
                    s = max(0, min(relative_s, sentence_count - 1))
                    return {"c": ci, "p": pi, "s": s,
                            "method": "paragraph_exact", "score": 0.9}

    # ---- 3. Fuzzy sentence similarity -------------------------------------
    best: dict[str, Any] | None = None
    anchor_len = len(anchor_norm)
    for idx, candidate_norm in enumerate(normalized_flat):
        if not candidate_norm:
            continue
        candidate_len = len(candidate_norm)
        if candidate_len > anchor_len * LENGTH_RATIO_WINDOW or (
            anchor_len > candidate_len * LENGTH_RATIO_WINDOW
        ):
            continue
        ratio = SequenceMatcher(None, anchor_norm, candidate_norm,
                                autojunk=False).ratio()
        if ratio < FUZZY_RATIO_MIN:
            continue
        score = 0.75 * ratio + 0.25 * context_score(idx)
        if best is None or score > best["score"]:
            c, p, s, _ = flat[idx]
            best = {"c": c, "p": p, "s": s, "method": "fuzzy_anchor",
                    "score": score, "ratio": ratio}

    if best and best["score"] >= FUZZY_ACCEPT_SCORE:
        return best
    return None


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def iso_from_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    """Return whether a process with the given PID exists (POSIX signal 0)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


# ---------------------------------------------------------------------------
# State repository
# ---------------------------------------------------------------------------

class StateStore:
    """On-disk versioned repository of book states and recent order."""

    def __init__(self, state_dir: str | None = None):
        self.state_dir = state_dir or config.PROGRESS_FILE_DIR
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, STATE_FILE_NAME)
        self.lock_path = os.path.join(self.state_dir, LOCK_FILE_NAME)
        self.backup_dir = os.path.join(self.state_dir, BACKUP_DIR_NAME)
        self.state: dict[str, Any] = self._read_disk()
        self._cleanup_temp_files()
        migrated = self.migrate_legacy()
        if migrated:
            logger.info("Legacy migration processed %d file(s)", migrated)

    # -- low level storage --------------------------------------------------

    def _blank_state(self) -> dict[str, Any]:
        return {
            "format": STATE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "revision": 0,
            "books": {},
            "recent": [],
            "legacy_migration": {},
            "migration_rollback": {"rolled_back": False, "source_hashes": []},
            "created_ns": time.time_ns(),
        }

    def _read_disk(self) -> dict[str, Any]:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return self._blank_state()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Quarantine rather than destroy an unparseable file.
            quarantine = os.path.join(
                self.state_dir,
                f"state.corrupt.{time.time_ns()}.json",
            )
            try:
                shutil.copy2(self.state_path, quarantine)
            except OSError:
                pass
            logger.error("State file was unparseable and quarantined: %s",
                         quarantine)
            return self._blank_state()

        if not isinstance(data, dict) or data.get("format") != STATE_FORMAT:
            logger.error("State file had an unknown format; starting fresh")
            return self._blank_state()
        data.setdefault("books", {})
        data.setdefault("recent", [])
        data.setdefault("legacy_migration", {})
        data.setdefault("migration_rollback",
                        {"rolled_back": False, "source_hashes": []})
        data.setdefault("revision", 0)
        return data

    def _cleanup_temp_files(self) -> None:
        """Remove crashed atomic-write temp files.

        Temp names embed the owning process PID (``.state.<pid>.<rand>.tmp``)
        so concurrent writers never unlink each other's *active* temp files:
        only files whose owning process is dead (or which are older than the
        safety age) are removed.
        """
        try:
            names = os.listdir(self.state_dir)
        except OSError:
            return
        now = time.time()
        for name in names:
            if not (name.startswith(".state.") and name.endswith(".tmp")):
                continue
            path = os.path.join(self.state_dir, name)
            match = re.match(r"^\.state\.(\d+)\..+\.tmp$", name)
            stale_by_age = False
            try:
                stale_by_age = (now - os.path.getmtime(path)) > STALE_TEMP_MAX_AGE
            except OSError:
                stale_by_age = True
            if match:
                owner_dead = not _pid_alive(int(match.group(1)))
                should_remove = owner_dead or stale_by_age
            else:
                # Pre-PID temp naming: only remove files that are certainly
                # leftovers from a much older session.
                should_remove = stale_by_age
            if should_remove:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _fsync_dir(self) -> None:
        try:
            fd = os.open(self.state_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _write_disk(self, state: dict[str, Any]) -> None:
        """Temp-file + fsync + atomic replace."""
        fd, tmp_path = tempfile.mkstemp(
            dir=self.state_dir,
            prefix=f".state.{os.getpid()}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_path)
            self._fsync_dir()
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _locked(self):
        return _FileLock(self.lock_path)

    # -- identity resolution ------------------------------------------------

    def _identity_match(self, identity: dict[str, Any]
                        ) -> tuple[dict[str, Any] | None, str | None]:
        """Find a stored book by EPUB identifier or exact content hash."""
        books = self.state["books"]
        if identity.get("epub_identifier"):
            for book_id, entry in books.items():
                stored_identity = entry.get("identity", {})
                if stored_identity.get("epub_identifier") == identity["epub_identifier"]:
                    return entry, "epub_identifier"
        for book_id, entry in books.items():
            stored_identity = entry.get("identity", {})
            if identity["content_hash"] in stored_identity.get(
                "known_content_hashes", []
            ) or stored_identity.get("content_hash") == identity["content_hash"]:
                return entry, "content_hash"
        return None, None

    def _path_match(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        for entry in self.state["books"].values():
            stored_identity = entry.get("identity", {})
            paths = stored_identity.get("known_paths", [])
            if stored_identity.get("canonical_path") == identity["canonical_path"] \
                    or identity["canonical_path"] in paths:
                return entry
        return None

    def resolve(self, identity: dict[str, Any],
                fingerprint: dict[str, Any] | None = None) -> dict[str, Any]:
        """Resolve the current file against stored state.

        Resolution order:

        1. exact identity hit (EPUB identifier / content hash),
        2. path match with changed content, confirmed by fingerprint,
        3. fuzzy fingerprint match over all stored books (file moved *and*
           edited),
        4. otherwise a brand new book; when a candidate exists but no anchor
           can be located, the old record is preserved and an explicit
           ``fallback`` result is returned instead of deleting progress.
        """
        entry, match_reason = self._identity_match(identity)
        identity_exact = entry is not None

        if entry is None:
            entry = self._path_match(identity)
            path_exact = entry is not None
        else:
            path_exact = False

        fuzzy_entry = None
        fuzzy_similarity = 0.0
        if entry is None and fingerprint is not None:
            for candidate in self.state["books"].values():
                stored_fingerprint = candidate.get("fingerprint")
                if not stored_fingerprint:
                    continue
                similarity = _multiset_jaccard(
                    fingerprint["paragraph_hashes"],
                    stored_fingerprint.get("paragraph_hashes", []),
                )
                if similarity > fuzzy_similarity:
                    fuzzy_similarity, fuzzy_entry = similarity, candidate
            if fuzzy_similarity >= FINGERPRINT_ACCEPT:
                entry = fuzzy_entry

        # No record at all -> new book.
        if entry is None:
            return {"status": "new", "book_id": identity["book_id"],
                    "entry": None, "position": None,
                    "reason": "no matching book in repository"}

        stored_position = entry.get("position") or {}

        # Exact identity: the file bytes are identical, so stored coordinates
        # only need structural validation/clamping.
        if identity_exact:
            c, p, s = stored_position.get("c", 0), stored_position.get("p", 0), \
                stored_position.get("s", 0)
            # Repository-level call with no document attached: trust the
            # stored coordinates of the byte-identical file.
            if self._current_chapters is None:
                return self._exact_result(entry, match_reason, c, p, s)
            if coordinates_valid(self._current_chapters, c, p, s):
                return self._exact_result(entry, match_reason, c, p, s)
            # Parser/structure drift: try the text anchor, then clamp.
            relocated = self._relocate_or_none(stored_position)
            if relocated is not None:
                return self._adopt(entry, identity, fingerprint, relocated,
                                   reason=f"anchor after {match_reason} drift")
            c, p, s = clamp_coordinates(self._current_chapters, c, p, s)
            return self._exact_result(entry, f"{match_reason}_clamped", c, p, s)

        # Same path but new content, or moved+edited fingerprint match.
        reason = "same path" if path_exact else \
            f"fingerprint {fuzzy_similarity:.2f}"
        relocated = self._relocate_or_none(stored_position)
        if relocated is None:
            # Explicit fallback: keep the old record, do not delete progress.
            old_identity = entry.get("identity", {})
            logger.warning(
                "Could not reliably relocate position for %s; "
                "keeping previous progress for %s",
                identity["canonical_path"],
                old_identity.get("canonical_path"),
            )
            return {
                "status": "fallback",
                "book_id": identity["book_id"],
                "entry": None,
                "position": None,
                "reason": reason,
                "kept_book_id": old_identity.get("book_id"),
                "kept_path": old_identity.get("canonical_path"),
            }
        return self._adopt(entry, identity, fingerprint, relocated, reason=reason)

    # Chapters of the document currently being resolved. Stored on the store
    # only for the duration of a resolve() call (set by the reader).
    _current_chapters: list[list[str]] | None = None

    def _relocate_or_none(self, stored_position: dict[str, Any]
                          ) -> dict[str, Any] | None:
        if self._current_chapters is None:
            return None
        return relocate_in_document(self._current_chapters, stored_position)

    def _exact_result(self, entry: dict[str, Any], reason: str | None,
                      c: int, p: int, s: int) -> dict[str, Any]:
        return {
            "status": "exact",
            "book_id": entry.get("identity", {}).get("book_id"),
            "entry": entry,
            "position": {"c": c, "p": p, "s": s},
            "reason": reason,
        }

    def _adopt(self, entry: dict[str, Any], identity: dict[str, Any],
               fingerprint: dict[str, Any] | None,
               relocated: dict[str, Any], reason: str) -> dict[str, Any]:
        """Adopt an old record under the current identity and persist it."""
        old_book_id = entry.get("identity", {}).get("book_id")
        new_book_id = identity["book_id"]

        adopted_position = dict(entry.get("position", {}))
        adopted_position.update({
            "c": relocated["c"],
            "p": relocated["p"],
            "s": relocated["s"],
        })
        adopted_position["relocated_from"] = {
            "method": relocated["method"],
            "score": round(relocated["score"], 4),
            "reason": reason,
            "at_ns": time.time_ns(),
        }

        if new_book_id == old_book_id:
            entry["identity"] = _merge_identity(entry.get("identity", {}),
                                                identity)
            entry["position"] = adopted_position
            if fingerprint is not None:
                entry["fingerprint"] = fingerprint
            adopted_entry = entry
        else:
            adopted_entry = dict(entry)
            adopted_entry["identity"] = _merge_identity(entry.get("identity", {}),
                                                        identity)
            adopted_entry["position"] = adopted_position
            if fingerprint is not None:
                adopted_entry["fingerprint"] = fingerprint
            self.state["books"].pop(old_book_id, None)
            self.state["books"][new_book_id] = adopted_entry
            self.state["recent"] = [
                new_book_id if book_id == old_book_id else book_id
                for book_id in self.state["recent"]
            ]

        self._commit_all("relocate")
        return {
            "status": "relocated",
            "book_id": new_book_id,
            "entry": adopted_entry,
            "position": {"c": relocated["c"], "p": relocated["p"],
                         "s": relocated["s"]},
            "reason": f"{reason} via {relocated['method']}",
        }

    # -- high level save ----------------------------------------------------

    def save_reading(self, *, identity: dict[str, Any],
                     chapters: list[list[str]], c: int, p: int, s: int,
                     scroll_offset: float, tts_enabled: bool,
                     auto_scroll_enabled: bool,
                     manual_scroll_anchor: tuple[int, int, int] | None = None,
                     playback_speed: float = 1.0,
                     completion_percentage: float = 0.0,
                     speed_reading_enabled: bool = False,
                     page_anchor: int | str | None = None) -> dict[str, Any]:
        """Persist one complete, versioned reading-state entry."""
        anchor = build_anchor(chapters, c, p, s)
        if anchor is None:
            return {"status": "error", "reason": "invalid position"}
        if page_anchor is not None:
            anchor["page_anchor"] = page_anchor
        fingerprint = build_fingerprint(chapters)
        now_ns = time.time_ns()

        incoming = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "fingerprint": fingerprint,
            "position": anchor,
            "ui": {
                "scroll_offset": float(scroll_offset),
                "manual_scroll_anchor": (
                    list(manual_scroll_anchor) if manual_scroll_anchor else None
                ),
                "tts_enabled": bool(tts_enabled),
                "auto_scroll_enabled": bool(auto_scroll_enabled),
                "speed_reading_enabled": bool(speed_reading_enabled),
                "playback_speed": float(playback_speed),
            },
            "completion_percentage": float(completion_percentage),
            "updated_ns": now_ns,
            "updated_at": iso_from_ns(now_ns),
        }
        return self._commit_entry(identity["book_id"], incoming)

    def _commit_all(self, change: str) -> None:
        """Persist the in-memory state (multi-entry changes) under the lock."""
        with self._locked():
            disk = self._read_disk()
            # Merge the books touched in memory; for untouched books keep disk.
            merged_books = dict(disk["books"])
            merged_books.update(self.state["books"])
            disk["books"] = merged_books
            disk["legacy_migration"].update(self.state.get("legacy_migration", {}))
            # Recent order from memory, but keep ids still present.
            recent = self.state["recent"]
            known = set(merged_books)
            disk["recent"] = [bid for bid in recent if bid in known]
            disk["revision"] = int(disk.get("revision", 0)) + 1
            disk["last_change"] = change
            self._write_disk(disk)
            self.state = disk

    def _commit_entry(self, book_id: str, incoming: dict[str, Any]) -> dict[str, Any]:
        with self._locked():
            disk = self._read_disk()
            existing = disk["books"].get(book_id)

            if existing is not None and \
                    int(existing.get("updated_ns", 0)) >= incoming["updated_ns"]:
                # A newer (or equal) version is already committed: this older
                # write must not silently replace it.
                self.state = disk
                return {
                    "status": "stale_rejected",
                    "revision": disk.get("revision", 0),
                    "committed_ns": existing.get("updated_ns"),
                    "incoming_ns": incoming["updated_ns"],
                }

            if existing is not None:
                incoming = _merge_book_entries(existing, incoming)

            disk["books"][book_id] = incoming
            disk["recent"] = [bid for bid in disk["recent"] if bid != book_id]
            disk["recent"].insert(0, book_id)
            disk["recent"] = disk["recent"][:RECENT_BOOKS_MAX]
            disk["revision"] = int(disk.get("revision", 0)) + 1
            disk["updated_ns"] = incoming["updated_ns"]
            self._write_disk(disk)
            self.state = disk
            return {"status": "committed",
                    "revision": disk["revision"]}

    # -- recent books -------------------------------------------------------

    def _existing_path(self, entry: dict[str, Any]) -> str | None:
        identity = entry.get("identity", {})
        candidates = []
        canonical = identity.get("canonical_path")
        if canonical:
            candidates.append(canonical)
        candidates.extend(identity.get("known_paths", []))
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def get_recent_books(self, limit: int = 5) -> list[dict[str, Any]]:
        """Recent books in repository order; same shape as the legacy menu."""
        result: list[dict[str, Any]] = []
        for book_id in self.state.get("recent", []):
            entry = self.state["books"].get(book_id)
            if entry is None:
                continue
            path = self._existing_path(entry)
            if path is None:
                continue
            identity = entry.get("identity", {})
            title = identity.get("title_hint") or os.path.splitext(
                os.path.basename(path)
            )[0]
            result.append({
                "title": title,
                "path": path,
                "percentage": entry.get("completion_percentage", 0.0),
            })
            if len(result) >= limit:
                break
        return result

    def find_most_recent_book(self) -> str | None:
        books = self.get_recent_books(limit=1)
        return books[0]["path"] if books else None

    # -- legacy migration ---------------------------------------------------

    def migrate_legacy(self, force: bool = False) -> int:
        """Scan for legacy title-named JSON files and migrate them once.

        Files skipped because the user rolled the migration back are only
        re-migrated when ``force`` is given (explicit re-migration).

        Returns the number of files migrated during this call.
        """
        import glob as _glob
        legacy_paths = sorted(
            _glob.glob(os.path.join(self.state_dir, LEGACY_GLOB_PATTERN))
        )
        if not legacy_paths:
            return 0

        if force:
            self.state["migration_rollback"] = {
                "rolled_back": False, "source_hashes": []
            }

        rollback_info = self.state.setdefault(
            "migration_rollback",
            {"rolled_back": False, "source_hashes": []},
        )
        rolled_back_hashes = set(rollback_info.get("source_hashes", [])) \
            if rollback_info.get("rolled_back") else set()

        migration_info = self.state.setdefault("legacy_migration", {})
        to_migrate: list[tuple[str, dict[str, Any]]] = []
        for legacy_path in legacy_paths:
            name = os.path.basename(legacy_path)
            try:
                source_hash = hash_file_content(legacy_path)
            except OSError:
                continue
            if source_hash in rolled_back_hashes:
                continue  # User rolled this file back; do not auto-re-migrate.
            known = migration_info.get(name)
            if known and known.get("source_hash") == source_hash:
                continue  # Already migrated and unchanged.
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            to_migrate.append((legacy_path, data))

        if not to_migrate:
            return 0

        for legacy_path, data in to_migrate:
            name = os.path.basename(legacy_path)
            self._migrate_one(legacy_path, name, data)

        # Rebuild recent order honoring preserved timestamps (old menu was
        # ordered by progress-file modification time).
        ordered = sorted(
            self.state["books"].values(),
            key=lambda entry: int(entry.get("updated_ns", 0)),
            reverse=True,
        )
        self.state["recent"] = [
            entry["identity"]["book_id"]
            for entry in ordered
            if "identity" in entry and "book_id" in entry["identity"]
        ][:RECENT_BOOKS_MAX]

        self._commit_all("legacy_migration")
        return len(to_migrate)

    def _migrate_one(self, legacy_path: str, name: str,
                     data: dict[str, Any]) -> None:
        original_path = data.get("original_file_path")
        source_mtime_ns = os.stat(legacy_path).st_mtime_ns
        source_hash = hash_file_content(legacy_path)

        # Keep a backup copy; the original file is never deleted.
        os.makedirs(self.backup_dir, exist_ok=True)
        backup_path = os.path.join(self.backup_dir, f"{name}.bak")
        if not os.path.exists(backup_path):
            shutil.copy2(legacy_path, backup_path)

        c = int(data.get("c", 0))
        p = int(data.get("p", 0))
        s = int(data.get("s", 0))
        ui_payload = {
            "scroll_offset": float(data.get("scroll_offset", 0)),
            "manual_scroll_anchor": data.get("manual_scroll_anchor"),
            "tts_enabled": bool(data.get("tts_enabled", True)),
            "auto_scroll_enabled": bool(data.get("auto_scroll_enabled", True)),
            "speed_reading_enabled": bool(data.get("speed_reading_enabled", False)),
            "playback_speed": float(data.get("playback_speed", 1.0)),
        }

        entry: dict[str, Any] | None = None
        book_id: str
        if original_path and os.path.exists(original_path):
            quiet_console = Console(quiet=True)
            chapters = content_parser.extract_content(original_path, quiet_console)
            if chapters and any(chapter for chapter in chapters):
                identity = compute_identity(original_path)
                c, p, s = clamp_coordinates(chapters, c, p, s)
                anchor = build_anchor(chapters, c, p, s)
                fingerprint = build_fingerprint(chapters)
                book_id = identity["book_id"]
                entry = {
                    "schema_version": SCHEMA_VERSION,
                    "identity": identity,
                    "fingerprint": fingerprint,
                    "position": anchor,
                    "ui": ui_payload,
                    "completion_percentage": float(
                        data.get("completion_percentage", 0.0)
                    ),
                    "migrated_from": name,
                }

        if entry is None:
            # Missing source file: preserve a stub with whatever we know.
            title = os.path.splitext(name)[0]
            title = re.sub(r"\.progress$", "", title)
            stub_identity = {
                "book_id": "b_stub_" + source_hash[:26],
                "canonical_path": original_path,
                "content_hash": None,
                "epub_identifier": None,
                "title_hint": title,
                "known_content_hashes": [],
                "known_paths": [original_path] if original_path else [],
                "source_missing": True,
            }
            book_id = stub_identity["book_id"]
            entry = {
                "schema_version": SCHEMA_VERSION,
                "identity": stub_identity,
                "fingerprint": None,
                "position": {"c": c, "p": p, "s": s},
                "ui": ui_payload,
                "completion_percentage": float(
                    data.get("completion_percentage", 0.0)
                ),
                "migrated_from": name,
            }

        entry["updated_ns"] = source_mtime_ns
        entry["updated_at"] = iso_from_ns(source_mtime_ns)
        self.state["books"][book_id] = entry
        self.state.setdefault("legacy_migration", {})[name] = {
            "book_id": book_id,
            "source_hash": source_hash,
            "backup_path": backup_path,
            "migrated_at_ns": time.time_ns(),
            "source_missing": bool(entry["identity"].get("source_missing")),
        }

    def rollback_migration(self) -> int:
        """Undo a legacy migration.

        Entries imported from legacy files are removed; the untouched legacy
        originals (and backups) remain, so the application returns to exactly
        its pre-migration state.  Returns the number of entries rolled back.
        """
        with self._locked():
            disk = self._read_disk()
            migration_info = disk.get("legacy_migration", {})
            if not migration_info:
                self.state = disk
                return 0
            book_ids = {info["book_id"] for info in migration_info.values()}
            source_hashes = [info["source_hash"]
                             for info in migration_info.values()]
            for book_id in book_ids:
                disk["books"].pop(book_id, None)
            disk["recent"] = [bid for bid in disk.get("recent", [])
                              if bid not in book_ids]
            disk["legacy_migration"] = {}
            # Persist the rollback so new instances do not auto-re-migrate.
            disk["migration_rollback"] = {
                "rolled_back": True,
                "source_hashes": source_hashes,
                "at_ns": time.time_ns(),
            }
            disk["revision"] = int(disk.get("revision", 0)) + 1
            self._write_disk(disk)
            self.state = disk
        return len(book_ids)


# ---------------------------------------------------------------------------
# Merging helpers
# ---------------------------------------------------------------------------

def _merge_identity(existing: dict[str, Any],
                    incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged.update({
        "book_id": incoming["book_id"],
        "canonical_path": incoming["canonical_path"],
        "content_hash": incoming["content_hash"],
        "size": incoming.get("size"),
        "mtime_ns": incoming.get("mtime_ns"),
        "title_hint": incoming.get("title_hint", merged.get("title_hint")),
    })
    if incoming.get("epub_identifier"):
        merged["epub_identifier"] = incoming["epub_identifier"]
    hashes = list(existing.get("known_content_hashes", []))
    for value in incoming.get("known_content_hashes", []):
        if value and value not in hashes:
            hashes.append(value)
    if incoming.get("content_hash") and incoming["content_hash"] not in hashes:
        hashes.append(incoming["content_hash"])
    paths = list(existing.get("known_paths", []))
    for value in incoming.get("known_paths", []):
        if value and value not in paths:
            paths.append(value)
    merged["known_content_hashes"] = hashes
    merged["known_paths"] = paths
    return merged


def _merge_book_entries(existing: dict[str, Any],
                        incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge a new commit over an existing entry.

    The incoming entry is newer, but durable identity aliases are unioned so
    observations from the older record are never lost.
    """
    merged = dict(incoming)
    merged["identity"] = _merge_identity(existing.get("identity", {}),
                                         incoming.get("identity", {}))
    provenance = existing.get("migrated_from")
    if provenance and "migrated_from" not in merged:
        merged["migrated_from"] = provenance
    return merged


# ---------------------------------------------------------------------------
# Advisory instance lock
# ---------------------------------------------------------------------------

class _FileLock:
    def __init__(self, lock_path: str, timeout: float = 10.0,
                 poll_interval: float = 0.02):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._file = None

    def __enter__(self):
        self._file = open(self.lock_path, "w")
        if fcntl is None:  # pragma: no cover - platform specific
            return self
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    # Do not block forever; revision comparison still applies.
                    logger.warning("Could not acquire instance lock in time")
                    return self
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            finally:
                self._file.close()
                self._file = None


# ---------------------------------------------------------------------------
# Module-level convenience API (default state directory)
# ---------------------------------------------------------------------------

_default_store: StateStore | None = None


def get_store(state_dir: str | None = None) -> StateStore:
    """Return the default store, or a store for an explicit directory."""
    global _default_store
    if state_dir is None:
        if _default_store is None:
            _default_store = StateStore(config.PROGRESS_FILE_DIR)
        return _default_store
    return StateStore(state_dir)


def get_recent_books(limit: int = 5) -> list[dict[str, Any]]:
    return get_store().get_recent_books(limit)


def find_most_recent_book() -> str | None:
    return get_store().find_most_recent_book()
