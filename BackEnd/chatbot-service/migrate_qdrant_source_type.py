"""
One-time migration: rename sourceType enum values in Qdrant.

Old values  → New values
-----------   ----------
"HR"        → "INTERNAL"
"CANDIDATE" → "EXTERNAL"

Applies to both CV and JD collections.
Uses scroll + set_payload (overwrite=False, keys=["sourceType"]) in batches
so vectors are never re-indexed — only the payload is patched.

Usage:
    cd BackEnd/chatbot-service
    python migrate_qdrant_source_type.py [--dry-run]
"""

import argparse
import sys
from typing import Optional
from qdrant_client.models import MatchAny

from dotenv import load_dotenv

load_dotenv(".env")

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SetPayload, PayloadSelector

from app.config import get_settings

settings = get_settings()

# Mapping of old → new sourceType values
_MIGRATIONS: dict[str, str] = {
    "HR":        "INTERNAL",
    "CANDIDATE": "EXTERNAL",
}

_POSITION_MIGRATIONS = [
    {
        "cv_ids": list(range(1, 11)),  # 1 -> 10
        "position": "Intern Java Developer",
    },
    {
        "cv_ids": list(range(12, 17)) + list(range(20, 25)),  # 12-16 + 20-24
        "position": "Fresher Java Developer",
    },
]

_BATCH_SIZE = 200  # Points fetched per scroll page


def _build_client() -> QdrantClient:
    if settings.QDRANT_USE_CLOUD:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=60)


def _migrate_collection(
    client: QdrantClient,
    collection: str,
    old_value: str,
    new_value: str,
    dry_run: bool,
) -> int:
    """
    Scroll through all points with sourceType == old_value and patch them to new_value.
    Returns the number of points updated.
    """
    scroll_filter = Filter(must=[
        FieldCondition(key="sourceType", match=MatchValue(value=old_value))
    ])

    updated = 0
    offset: Optional[str] = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=_BATCH_SIZE,
            offset=offset,
            with_payload=False,   # payload not needed — we only need IDs
            with_vectors=False,
        )

        if not results:
            break

        point_ids = [r.id for r in results]

        if dry_run:
            print(
                f"  [DRY-RUN] {collection}: would patch {len(point_ids)} point(s) "
                f"'{old_value}' → '{new_value}'"
            )
        else:
            client.set_payload(
                collection_name=collection,
                payload={"sourceType": new_value},
                points=point_ids,
            )
            print(
                f"  [PATCHED] {collection}: {len(point_ids)} point(s) "
                f"'{old_value}' → '{new_value}'"
            )

        updated += len(point_ids)

        if next_offset is None:
            break
        offset = next_offset

    return updated


def _update_positions(
    client: QdrantClient,
    collection: str,
    dry_run: bool,
) -> int:
    updated = 0

    for rule in _POSITION_MIGRATIONS:
        cv_ids = rule["cv_ids"]
        position = rule["position"]

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="cvId",
                    match=MatchAny(any=cv_ids)
                )
            ]
        )

        results, _ = client.scroll(
            collection_name=collection,
            scroll_filter=search_filter,
            limit=100,
            with_payload=False,
            with_vectors=False,
        )

        if not results:
            print(f"  [SKIP] No CV found for ids={cv_ids}")
            continue

        point_ids = [r.id for r in results]

        if dry_run:
            print(
                f"  [DRY-RUN] {collection}: would update {len(point_ids)} point(s) "
                f"to position='{position}'"
            )
        else:
            client.set_payload(
                collection_name=collection,
                payload={"position": position},
                points=point_ids,
            )
            print(
                f"  [PATCHED] {collection}: updated {len(point_ids)} point(s) "
                f"position='{position}'"
            )

        updated += len(point_ids)

    return updated

def run(dry_run: bool) -> None:
    client = _build_client()
    collections = [settings.CV_COLLECTION_NAME]

    total = 0
    for collection in collections:
        print(f"\n[Migration] Processing collection: '{collection}'")

        for old_val, new_val in _MIGRATIONS.items():
            count = _migrate_collection(client, collection, old_val, new_val, dry_run)
            if count == 0:
                print(f"  [SKIP] No points with sourceType='{old_val}' found.")
            total += count

        position_count = _update_positions(client, collection, dry_run)
        total += position_count

    action = "would update" if dry_run else "updated"
    print(f"\n[Migration] Done. Total points {action}: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Qdrant sourceType values.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing any changes.",
    )
    args = parser.parse_args()

    try:
        run(dry_run=args.dry_run)
    except Exception as exc:
        print(f"[Migration ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
