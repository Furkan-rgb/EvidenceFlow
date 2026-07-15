"""Business persistence, durable local jobs, and resume concurrency guards."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel

from app.errors import InvalidReviewDecisionError, ReviewNotResumableError

JsonDict = dict[str, Any]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _payload(value: object) -> JsonDict:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError("Persistence payloads must be mappings or Pydantic models")


class SQLiteReviewRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._migrations_dir = Path(__file__).with_name("migrations")

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.database_path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA journal_mode = WAL")
        await connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            await connection.close()

    async def migrate(self) -> None:
        async with self.connect() as connection:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            async with connection.execute("SELECT version FROM schema_migrations") as cursor:
                applied = {str(row[0]) for row in await cursor.fetchall()}
            for migration in sorted(self._migrations_dir.glob("*.sql")):
                if migration.name in applied:
                    continue
                await connection.executescript(migration.read_text(encoding="utf-8"))
                await connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (migration.name, _now()),
                )
            await connection.commit()

    async def health(self) -> bool:
        try:
            async with self.connect() as connection, connection.execute(
                "SELECT 1"
            ) as cursor:
                return (await cursor.fetchone()) is not None
        except aiosqlite.Error:
            return False

    async def create(
        self,
        review_id: str,
        thread_id: str,
        documents: Sequence[dict[str, object]],
    ) -> None:
        timestamp = _now()
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "INSERT INTO reviews(review_id, thread_id, status, created_at, updated_at) "
                "VALUES (?, ?, 'processing', ?, ?)",
                (review_id, thread_id, timestamp, timestamp),
            )
            for document in documents:
                await connection.execute(
                    "INSERT INTO documents(document_id, review_id, filename, artifact_id, "
                    "sha256, size_bytes, content_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document["document_id"],
                        review_id,
                        document["filename"],
                        document["artifact_id"],
                        document["sha256"],
                        document["size_bytes"],
                        document["content_type"],
                        timestamp,
                    ),
                )
            await self._insert_job(connection, review_id, "start", {})
            await self._event(
                connection, review_id, "review_created", {"documents": len(documents)}
            )
            await connection.commit()

    async def get(self, review_id: str) -> JsonDict | None:
        async with self.connect() as connection:
            # The review row, pending items, documents, and report form one aggregate.
            # Keep them on a single read snapshot so a concurrent resume cannot expose
            # the old ``needs_review`` status alongside newly decided review items.
            await connection.execute("BEGIN")
            async with connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?", (review_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            result = dict(row)
            result["snapshot"] = json.loads(result.pop("snapshot_json") or "{}")
            result["error"] = json.loads(result.pop("error_json") or "null")
            async with connection.execute(
                "SELECT document_id, filename, artifact_id, sha256, size_bytes, content_type "
                "FROM documents WHERE review_id = ? ORDER BY created_at, document_id",
                (review_id,),
            ) as cursor:
                result["documents"] = [dict(item) for item in await cursor.fetchall()]
            async with connection.execute(
                "SELECT payload_json FROM review_items WHERE review_id = ? AND state = 'pending' "
                "ORDER BY created_at, review_item_id",
                (review_id,),
            ) as cursor:
                result["pending_reviews"] = [
                    json.loads(item[0]) for item in await cursor.fetchall()
                ]
            async with connection.execute(
                "SELECT report_json, markdown FROM reports WHERE review_id = ?", (review_id,)
            ) as cursor:
                report_row = await cursor.fetchone()
            result["report"] = json.loads(report_row[0]) if report_row else None
            result["report_markdown"] = report_row[1] if report_row else None
            return result

    async def get_document(self, review_id: str, document_id: str) -> JsonDict | None:
        async with self.connect() as connection:
            async with connection.execute(
                "SELECT * FROM documents WHERE review_id = ? AND document_id = ?",
                (review_id, document_id),
            ) as cursor:
                row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_status(self, review_id: str, status: str, **values: object) -> None:
        assignments = ["status = ?", "updated_at = ?", "revision = revision + 1"]
        parameters: list[object] = [status, _now()]
        for key in ("report_status", "snapshot", "error"):
            if key not in values:
                continue
            column = f"{key}_json" if key in {"snapshot", "error"} else key
            assignments.append(f"{column} = ?")
            parameters.append(_json(values[key]) if key in {"snapshot", "error"} else values[key])
        parameters.append(review_id)
        async with self.connect() as connection:
            await connection.execute(
                f"UPDATE reviews SET {', '.join(assignments)} WHERE review_id = ?", parameters
            )
            await self._event(connection, review_id, f"status_{status}", values)
            await connection.commit()

    async def save_snapshot(self, review_id: str, snapshot: object) -> None:
        async with self.connect() as connection:
            await connection.execute(
                "UPDATE reviews SET snapshot_json = ?, updated_at = ?, revision = revision + 1 "
                "WHERE review_id = ?",
                (_json(snapshot), _now(), review_id),
            )
            await connection.commit()

    async def save_review_items(self, review_id: str, items: Sequence[object]) -> None:
        timestamp = _now()
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "DELETE FROM review_items WHERE review_id = ? AND state = 'pending'", (review_id,)
            )
            for raw in items:
                payload = _payload(raw)
                item_id = str(payload.get("review_item_id") or payload.get("item_id"))
                item_type = str(payload.get("type") or payload.get("item_type"))
                await connection.execute(
                    "INSERT INTO review_items(review_item_id, review_id, item_type, "
                    "state, fingerprint, payload_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
                    (
                        item_id,
                        review_id,
                        item_type,
                        payload.get("fingerprint"),
                        _json(payload),
                        timestamp,
                        timestamp,
                    ),
                )
            await connection.execute(
                "UPDATE reviews SET status = 'needs_review', updated_at = ?, "
                "revision = revision + 1 "
                "WHERE review_id = ?",
                (timestamp, review_id),
            )
            await self._event(connection, review_id, "review_items_created", {"count": len(items)})
            await connection.commit()

    async def save_review_decisions(self, review_id: str, decisions: list[object]) -> None:
        await self.begin_resume(review_id, decisions)

    async def begin_resume(self, review_id: str, decisions: Sequence[object]) -> str:
        timestamp = _now()
        payloads = [_payload(decision) for decision in decisions]
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT status FROM reviews WHERE review_id = ?", (review_id,)
            ) as cursor:
                review = await cursor.fetchone()
            if review is None or review[0] != "needs_review":
                await connection.rollback()
                raise ReviewNotResumableError("Review is not waiting for human decisions")
            async with connection.execute(
                "SELECT review_item_id FROM review_items WHERE review_id = ? AND state = 'pending'",
                (review_id,),
            ) as cursor:
                pending = {str(row[0]) for row in await cursor.fetchall()}
            provided = {str(item.get("review_item_id")) for item in payloads}
            if pending != provided or len(provided) != len(payloads):
                await connection.rollback()
                raise InvalidReviewDecisionError(
                    "Provide exactly one decision for every pending review item",
                    details={"pending": sorted(pending), "provided": sorted(provided)},
                )
            for item in payloads:
                item_id = str(item["review_item_id"])
                item.setdefault("decision_id", f"decision-{uuid4().hex}")
                item.setdefault("decided_at", timestamp)
                await connection.execute(
                    "INSERT INTO review_decisions(decision_id, review_id, review_item_id, action, "
                    "value_json, selected_field_id, actor, decided_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item["decision_id"],
                        review_id,
                        item_id,
                        item["action"],
                        _json(item.get("value")) if "value" in item else None,
                        item.get("selected_field_id"),
                        item.get("actor", "local_reviewer"),
                        item["decided_at"],
                    ),
                )
                await connection.execute(
                    "UPDATE review_items SET state = 'decided', updated_at = ? "
                    "WHERE review_id = ? AND review_item_id = ?",
                    (timestamp, review_id, item_id),
                )
            await connection.execute(
                "UPDATE reviews SET status = 'processing', resume_count = resume_count + 1, "
                "updated_at = ?, revision = revision + 1 WHERE review_id = ?",
                (timestamp, review_id),
            )
            job_id = await self._insert_job(
                connection, review_id, "resume", {"decisions": payloads}
            )
            await self._event(connection, review_id, "review_resumed", {"count": len(payloads)})
            await connection.commit()
            return job_id

    async def save_report(self, review_id: str, report: object, markdown: str) -> None:
        report_payload = _payload(report)
        timestamp = _now()
        async with self.connect() as connection:
            await connection.execute(
                "INSERT INTO reports(review_id, report_json, markdown, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(review_id) DO UPDATE SET report_json = excluded.report_json, "
                "markdown = excluded.markdown, created_at = excluded.created_at",
                (review_id, _json(report_payload), markdown, timestamp),
            )
            await connection.execute(
                "UPDATE reviews SET status = 'completed', report_status = ?, updated_at = ?, "
                "revision = revision + 1 WHERE review_id = ?",
                (report_payload.get("status"), timestamp, review_id),
            )
            await self._event(connection, review_id, "report_saved", {})
            await connection.commit()

    async def claim_next_job(self) -> JsonDict | None:
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT * FROM workflow_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await connection.rollback()
                return None
            timestamp = _now()
            await connection.execute(
                "UPDATE workflow_jobs SET status = 'running', attempts = attempts + 1, "
                "updated_at = ? "
                "WHERE job_id = ?",
                (timestamp, row["job_id"]),
            )
            await connection.commit()
            result = dict(row)
            result["payload"] = json.loads(result.pop("payload_json") or "{}")
            result["status"] = "running"
            return result

    async def finish_job(self, job_id: str, *, failed: bool = False) -> None:
        async with self.connect() as connection:
            await connection.execute(
                "UPDATE workflow_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                ("failed" if failed else "completed", _now(), job_id),
            )
            await connection.commit()

    async def recover_jobs(self) -> int:
        async with self.connect() as connection:
            cursor = await connection.execute(
                "UPDATE workflow_jobs SET status = 'queued', kind = 'recover', updated_at = ? "
                "WHERE status = 'running'",
                (_now(),),
            )
            await connection.commit()
            return cursor.rowcount

    async def _insert_job(
        self, connection: aiosqlite.Connection, review_id: str, kind: str, payload: object
    ) -> str:
        job_id = f"job_{uuid4().hex}"
        timestamp = _now()
        await connection.execute(
            "INSERT INTO workflow_jobs(job_id, review_id, kind, status, payload_json, created_at, "
            "updated_at) VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (job_id, review_id, kind, _json(payload), timestamp, timestamp),
        )
        return job_id

    async def _event(
        self, connection: aiosqlite.Connection, review_id: str, event_type: str, payload: object
    ) -> None:
        await connection.execute(
            "INSERT INTO review_events(review_id, event_type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (review_id, event_type, _json(payload), _now()),
        )
