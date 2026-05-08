import json
import asyncio
import time
import hashlib
from datetime import datetime
from typing import List

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import aio_pika
from aio_pika import ExchangeType

from app.config import get_settings
from app.services.embedding import embedding_service
from app.services.qdrant import qdrant_service

settings = get_settings()

# ============================================================
# JD CHUNKED FLOW — Consumer for JD chunk embedding
# Aligned with Java RabbitMQConfig.JD_CHUNKED_* constants.
# ============================================================

JD_CHUNKED_QUEUE           = "jd.chunked.queue"
JD_CHUNKED_DLQ             = "jd.chunked.queue.dlq"
JD_CHUNKED_EXCHANGE        = "jd.chunked.exchange"
JD_CHUNKED_ROUTING_KEY     = "jd.chunked"
JD_CHUNKED_DLQ_ROUTING_KEY = "jd.chunked.dlq"

JD_EMBED_REPLY_QUEUE       = "jd.embed.reply.queue"


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def _validate_jd_chunked_event(event: dict) -> None:
    """
    Validate the JDChunkedEvent payload sent by Java PositionService.
    Raises ValueError for any structural or content violation so the
    message is nack'd immediately and routes to DLQ without retrying.
    """
    required_top = ["positionId", "positionTitle", "chunks", "totalChunks"]
    for field in required_top:
        if field not in event:
            raise ValueError(f"Missing required field: '{field}'")

    position_id = event["positionId"]
    if not isinstance(position_id, int) or position_id <= 0:
        raise ValueError(f"Invalid positionId: {position_id}")

    chunks = event["chunks"]
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError(f"chunks must be a non-empty list, got: {chunks}")

    total_chunks = event["totalChunks"]
    if not isinstance(total_chunks, int) or total_chunks != len(chunks):
        raise ValueError(
            f"totalChunks mismatch: declared={total_chunks}, actual={len(chunks)}"
        )

    # Spot-check first chunk structure
    sample = chunks[0]
    required_chunk_fields = ["positionId", "chunkIndex", "sectionName", "chunkText"]
    for field in required_chunk_fields:
        if field not in sample:
            raise ValueError(f"Chunk missing required field: '{field}'")


# ------------------------------------------------------------
# Core embedding logic
# ------------------------------------------------------------

async def _embed_jd_chunks(event: dict) -> None:
    """
    Embeds all JD chunks from a single JDChunkedEvent and upserts them
    as individual Qdrant points.

    Strategy (Small-to-Big):
      - Each chunk gets its own vector point keyed as 'jd_{id}_chunk_{idx}'.
      - All points share the same positionId payload field — the RAG retriever
        de-duplicates by positionId, then fetches the full JD text via the
        Java internal API instead of reading it from Qdrant.
    """
    start_time = time.time()

    position_id   = event["positionId"]
    position_title = event.get("positionTitle", "")
    seniority      = event.get("seniority", "")
    chunks: List[dict] = event["chunks"]

    print(f"[JD] Processing {len(chunks)} chunks for position {position_id} ({position_title})...")

    # 1. Batch-embed all chunk texts in one pass (efficient GPU/CPU usage)
    chunk_texts = [c["chunkText"] for c in chunks]
    if not all(isinstance(t, str) and t.strip() for t in chunk_texts):
        raise ValueError(f"[JD] One or more chunks have empty/invalid chunkText for position {position_id}")

    print(f"[JD] Embedding {len(chunk_texts)} chunk texts...")
    embeddings = embedding_service.embed_batch(chunk_texts, show_progress=True)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"[JD] Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}"
        )

    # 2. Delete all stale points for this position before inserting the new set.
    #    This handles the JD-update scenario cleanly without orphan vectors.
    print(f"[JD] Deleting old chunk embeddings for position {position_id}...")
    try:
        delete_filter = Filter(
            must=[FieldCondition(key="positionId", match=MatchValue(value=position_id))]
        )
        qdrant_service.delete_by_filter(
            collection_name=settings.JD_COLLECTION_NAME,
            filters=delete_filter
        )
        print(f"[JD] Old embeddings deleted for position {position_id}")
    except Exception as e:
        # Non-fatal — warn and continue; worst case we get duplicate vectors
        print(f"[JD] Warning: Failed to delete old embeddings for position {position_id}: {e}")

    # 3. Build PointStruct list — one point per chunk
    version = int(datetime.now().timestamp())
    points: List[PointStruct] = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_index = chunk.get("chunkIndex", 0)
        # Deterministic integer ID derived from position + chunk index + version
        point_id_str = f"jd_{position_id}_chunk_{chunk_index}_v{version}"
        point_id = int(hashlib.md5(point_id_str.encode()).hexdigest(), 16) % (2 ** 63)

        payload = {
            # Parent identifiers — used for Small-to-Big lookup by chatbot-service
            "positionId":    position_id,
            "positionTitle": position_title,
            "seniority":     seniority,

            # Chunk metadata
            "sectionName":    chunk.get("sectionName", ""),
            "chunkIndex":     chunk_index,
            "chunkText":      chunk.get("chunkText", ""),

            # Stats
            "words":          chunk.get("words", 0),
            "tokensEstimate": chunk.get("tokensEstimate", 0),

            # Versioning
            "version":   version,
            "createdAt": datetime.now().isoformat(),
        }

        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    # 4. Upsert in batches to Qdrant
    print(f"[JD] Upserting {len(points)} chunk points to Qdrant collection '{settings.JD_COLLECTION_NAME}'...")
    batch_size = settings.BATCH_SIZE

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        success = qdrant_service.upsert_points(
            collection_name=settings.JD_COLLECTION_NAME,
            points=batch
        )
        if not success:
            batch_num = i // batch_size + 1
            raise RuntimeError(f"[JD] Failed to upsert batch {batch_num} for position {position_id}")

        batch_num = i // batch_size + 1
        total_batches = (len(points) - 1) // batch_size + 1
        print(f"[JD] Upserted batch {batch_num}/{total_batches}")

    elapsed = time.time() - start_time
    print(f"[JD] Successfully embedded position {position_id} — {len(points)} chunks in {elapsed:.2f}s")


# ------------------------------------------------------------
# Message handler
# ------------------------------------------------------------

async def _process_jd_message(message: aio_pika.IncomingMessage) -> None:
    """
    Handle a single JDChunkedEvent message.
    - Validation failure  → nack (no requeue) → DLQ immediately.
    - Processing failure  → nack with retry counter up to max_retries.
    - Success             → ack.
    """
    try:
        event = json.loads(message.body.decode())
        position_id = event.get("positionId", "?")
        total = event.get("totalChunks", "?")
        print(f"\n{'='*60}")
        print(f"[JD] Received JDChunkedEvent: positionId={position_id}, totalChunks={total}")
        print(f"{'='*60}")

    except json.JSONDecodeError as e:
        print(f"[JD] Invalid JSON in message body: {e}")
        await message.nack(requeue=False)
        return

    headers     = message.headers or {}
    retry_count = headers.get("x-retry-count", 0)
    max_retries = 3

    try:
        _validate_jd_chunked_event(event)
    except ValueError as e:
        # Structural error — dead-letter immediately, no point retrying
        print(f"[JD] Validation failed (no retry): {e}")
        await message.nack(requeue=False)
        return

    try:
        await _embed_jd_chunks(event)
        await message.ack()
        print(f"[JD] Acked message for position {event['positionId']}\n")
        
        # Publish success reply to Java
        await _publish_reply(
            position_id=event['positionId'],
            batch_id=event.get('batchId'),
            success=True
        )

    except Exception as e:
        print(f"[JD] Processing error (attempt {retry_count + 1}/{max_retries}): {e}")
        import traceback
        traceback.print_exc()

        # Publish failure reply to Java on max retries
        if retry_count >= max_retries - 1:
            await _publish_reply(
                position_id=event.get('positionId', 0),
                batch_id=event.get('batchId'),
                success=False,
                error_msg=str(e)
            )

        if retry_count < max_retries:
            print(f"[JD] Nacking for retry (count={retry_count + 1})...")
        else:
            print(f"[JD] Max retries reached, routing to DLQ.")

        # Always nack without requeue — DLX on the queue handles retry routing
        await message.nack(requeue=False)

async def _publish_reply(position_id: int, batch_id: str, success: bool, error_msg: str = None) -> None:
    """Publish the final result (success/failure) back to Java via jd.embed.reply.queue."""
    try:
        connection = await aio_pika.connect_robust(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            login=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASSWORD,
        )
        async with connection:
            channel = await connection.channel()
            reply_event = {
                "cvId": position_id, # Reusing cvId as positionId for the generic EmbedReplyEvent payload
                "success": success,
                "errorMessage": error_msg,
                "batchId": batch_id
            }
            await channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(reply_event).encode()),
                routing_key=JD_EMBED_REPLY_QUEUE
            )
            status_str = "SUCCESS" if success else "FAILED"
            print(f"[JD] Published reply to {JD_EMBED_REPLY_QUEUE} -> posId={position_id}, batchId={batch_id}, status={status_str}")
    except Exception as e:
        print(f"[JD] FATAL: Could not publish reply for posId={position_id}: {e}")


# ------------------------------------------------------------
# Consumer bootstrap
# ------------------------------------------------------------

async def consume_jd_chunked_events() -> None:
    """Connect to RabbitMQ and start consuming jd.chunked.queue (async)."""
    try:
        connection = await aio_pika.connect_robust(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            login=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASSWORD,
            heartbeat=600,
        )

        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)

            exchange = await channel.declare_exchange(
                JD_CHUNKED_EXCHANGE, ExchangeType.DIRECT, durable=True
            )

            # Declare DLQ first so the main queue can reference it
            dlq = await channel.declare_queue(JD_CHUNKED_DLQ, durable=True)
            await dlq.bind(exchange, routing_key=JD_CHUNKED_DLQ_ROUTING_KEY)

            # Declare main queue with DLX arguments
            queue = await channel.declare_queue(
                JD_CHUNKED_QUEUE,
                durable=True,
                arguments={
                    "x-dead-letter-exchange":    JD_CHUNKED_EXCHANGE,
                    "x-dead-letter-routing-key": JD_CHUNKED_DLQ_ROUTING_KEY,
                },
            )
            await queue.bind(exchange, routing_key=JD_CHUNKED_ROUTING_KEY)

            print("[JD] Embedding Worker started (Async)")
            print(f"[JD] RabbitMQ: {settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}")
            print(f"[JD] Listening on queue : {JD_CHUNKED_QUEUE}")
            print(f"[JD] Dead-letter queue  : {JD_CHUNKED_DLQ}\n")
            print("[JD] Waiting for messages. Press Ctrl+C to exit...\n")

            await queue.consume(_process_jd_message)
            await asyncio.Future()  # run forever

    except asyncio.CancelledError:
        print("\n[JD] Shutting down JD worker...")
    except Exception as e:
        print(f"[JD] Worker fatal error: {e}")
        import traceback
        traceback.print_exc()


def start_jd_consumer() -> None:
    """Entry point called by worker_jd.py."""
    try:
        asyncio.run(consume_jd_chunked_events())
    except KeyboardInterrupt:
        print("\n[JD] Gracefully shutting down...")
