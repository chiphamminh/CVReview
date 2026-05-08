import pika
import json
import asyncio
import time
from datetime import datetime
import hashlib
from qdrant_client.models import PointStruct
import aio_pika
from aio_pika import ExchangeType

from app.config import get_settings
from app.services.embedding import embedding_service
from app.services.qdrant import qdrant_service

settings = get_settings()

# ============================================================
# CV EMBED FLOW - Phase 3 Two-Stage Pipeline (Java Chunking)
# ============================================================

CV_EMBED_QUEUE = "cv.embed.queue"
CV_EMBED_DLQ = "cv.embed.queue.dlq"
CV_EMBED_EXCHANGE = "cv.embed.exchange"
CV_EMBED_ROUTING_KEY = "cv.embed"
CV_EMBED_DLQ_ROUTING_KEY = "cv.embed.dlq"

CV_EMBED_REPLY_QUEUE = "cv.embed.reply.queue"

def validate_cv_chunked_event(event: dict) -> None:
    """Validate CV chunked event payload"""
    required_fields = ['cvId', 'chunks', 'totalChunks']
    for field in required_fields:
        if field not in event:
            raise ValueError(f"Missing required field: {field}")
    
    cv_id = event['cvId']
    if not isinstance(cv_id, int) or cv_id <= 0:
        raise ValueError(f"Invalid cvId: {cv_id}")
    
    chunks = event['chunks']
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError(f"chunks is empty or not a list for CV {cv_id}")

async def publish_reply(channel: aio_pika.Channel, reply_event: dict):
    """Publish reply to cv.embed.reply.queue via default exchange"""
    exchange = channel.default_exchange
    message = aio_pika.Message(
        body=json.dumps(reply_event).encode(),
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
    )
    await exchange.publish(
        message,
        routing_key=CV_EMBED_REPLY_QUEUE
    )
    print(f"Published reply to {CV_EMBED_REPLY_QUEUE}: {reply_event['success']}")

async def embed_cv_from_event(event: dict, channel: aio_pika.Channel):
    """
    Embed CV từ RabbitMQ event Phase 3 (với chunking làm từ Java)
    """
    start_time = time.time()
    
    try:
        validate_cv_chunked_event(event)
    except ValueError as e:
        print(f"Invalid CV event payload: {e}")
        raise
    
    cv_id = event['cvId']
    chunks = event['chunks']
    batch_id = event.get('batchId')
    
    reply_event = {
        "cvId": cv_id,
        "batchId": batch_id,
        "success": False,
        "errorMessage": None
    }
    
    try:
        print(f"Processing CV embedding for CV {cv_id}...")
        print(f"Total chunks from Java: {len(chunks)}")
        
        # 1. Extract chunk texts
        chunk_texts = [chunk['chunkText'] for chunk in chunks]
        
        # Validate chunk texts
        if not all(isinstance(text, str) and text.strip() for text in chunk_texts):
            raise ValueError("Some chunks have empty or invalid text")
        
        # 2. Embed all chunks in batch
        embeddings = embedding_service.embed_batch(chunk_texts, show_progress=False)
        
        if len(embeddings) != len(chunks):
            raise Exception(f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}")
        
        # 3. Delete old embeddings for this CV (nếu có)
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            delete_filter = Filter(must=[FieldCondition(key="cvId", match=MatchValue(value=cv_id))])
            qdrant_service.delete_by_filter(collection_name=settings.CV_COLLECTION_NAME, filters=delete_filter)
        except Exception as e:
            print(f"Warning: Failed to delete old embeddings: {e}")
        
        # 4. Prepare points
        version = int(datetime.now().timestamp())
        points = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_index = chunk.get('chunkIndex', 0)
            point_id_str = f"cv_{cv_id}_chunk_{chunk_index}_v{version}"
            point_id = int(hashlib.md5(point_id_str.encode()).hexdigest(), 16) % (2**63)
            
            source_type = chunk.get("sourceType", "")
            position_id_val = chunk.get("positionId")
            
            applied_position_ids = [position_id_val] if source_type == "EXTERNAL" and position_id_val is not None else []
            
            payload = {
                "cvId": cv_id,
                "candidateId": chunk.get("candidateId"),
                "hrId": chunk.get("hrId"),
                "positionId": position_id_val,
                "position": chunk.get("position", ""),
                "applied_position_ids": applied_position_ids,
                "section": chunk.get("section", ""),
                "chunkIndex": chunk_index,
                "chunkText": chunk.get("chunkText", ""),
                "skills": chunk.get("skills", []),
                "experienceYears": chunk.get("experienceYears"),
                "seniorityLevel": chunk.get("seniorityLevel", "Unknown"),
                "email": chunk.get("email", ""),
                "companies": chunk.get("companies", []),
                "degrees": chunk.get("degrees", []),
                "dateRanges": chunk.get("dateRanges", []),
                "version": version,
                "is_latest": True,
                "createdAt": chunk.get("createdAt", datetime.now().isoformat()),
                "words": chunk.get("words", 0),
                "tokensEstimate": chunk.get("tokensEstimate", 0),
                "cvStatus": "EMBEDDED",
                "sourceType": source_type,
            }
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        
        # 5. Upsert to Qdrant in batches
        batch_size = settings.BATCH_SIZE
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            success = qdrant_service.upsert_points(
                collection_name=settings.CV_COLLECTION_NAME,
                points=batch
            )
            if not success:
                raise Exception(f"Failed to upsert batch {i//batch_size + 1}")
        
        processing_time = time.time() - start_time
        print(f"Successfully embedded CV {cv_id} ({len(points)} chunks) in {processing_time:.2f}s")
        
        reply_event["success"] = True
        await publish_reply(channel, reply_event)
        
    except Exception as e:
        print(f"Error embedding CV {cv_id}: {e}")
        reply_event["success"] = False
        reply_event["errorMessage"] = str(e)
        await publish_reply(channel, reply_event)
        raise

async def process_cv_message(message: aio_pika.IncomingMessage, channel: aio_pika.Channel):
    """Process a single CV message with retry logic"""
    try:
        event = json.loads(message.body.decode())
        print(f"\n{'='*60}")
        print(f"Received CV event: CV ID {event.get('cvId')}")
        print(f"{'='*60}")
        
        headers = message.headers or {}
        retry_count = headers.get('x-retry-count', 0)
        max_retries = 3
        
        try:
            await embed_cv_from_event(event, channel)
            await message.ack()
            print(f"Acknowledged message for CV {event.get('cvId')}\n")
            
        except Exception as e:
            print(f"Error processing CV event (attempt {retry_count + 1}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                await message.nack(requeue=False)
            else:
                await message.nack(requeue=False)
            
            import traceback
            traceback.print_exc()
            
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in CV message: {e}")
        await message.nack(requeue=False)
    except Exception as e:
        print(f"Unexpected error in CV message processing: {e}")
        await message.nack(requeue=False)

async def consume_cv_chunked_events():
    """Consume CV events from RabbitMQ (Async)"""
    try:
        connection = await aio_pika.connect_robust(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            login=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASSWORD,
            heartbeat=600
        )
        
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)
            
            exchange = await channel.declare_exchange(
                CV_EMBED_EXCHANGE,
                ExchangeType.DIRECT,
                durable=True
            )
            
            dlq = await channel.declare_queue(CV_EMBED_DLQ, durable=True)
            await dlq.bind(exchange, routing_key=CV_EMBED_DLQ_ROUTING_KEY)
            
            queue = await channel.declare_queue(
                CV_EMBED_QUEUE,
                durable=True,
                arguments={
                    'x-dead-letter-exchange': CV_EMBED_EXCHANGE,
                    'x-dead-letter-routing-key': CV_EMBED_DLQ_ROUTING_KEY,
                }
            )

            await queue.bind(exchange, routing_key=CV_EMBED_ROUTING_KEY)
            
            print('CV Embedding Worker started (Async - Java Chunking Restored)')
            print(f'Connected to RabbitMQ: {settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}')
            print(f'Listening on queue: {CV_EMBED_QUEUE}')
            print(f'Dead letter queue: {CV_EMBED_DLQ}\n')
            print('Waiting for CV messages. Press Ctrl+C to exit...\n')
            
            async def on_message(msg):
                await process_cv_message(msg, channel)
                
            await queue.consume(on_message)
            await asyncio.Future()
            
    except asyncio.CancelledError:
        print('\nShutting down CV worker...')
    except Exception as e:
        print(f"CV Worker error: {e}")
        import traceback
        traceback.print_exc()

def start_cv_consumer():
    """Entry point to start the consumer"""
    try:
        asyncio.run(consume_cv_chunked_events())
    except KeyboardInterrupt:
        print('\nGracefully shutting down...')
