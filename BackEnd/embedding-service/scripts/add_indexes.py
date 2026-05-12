import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

"""
Fix Qdrant indexes với đúng field types
Chạy script này để tạo lại indexes
"""
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType
from app.config import get_settings

settings = get_settings()

def create_indexes():
    """Create proper indexes for CV and JD collections"""
    
    # Connect to Qdrant
    if settings.QDRANT_USE_CLOUD:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
    else:
        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
    
    print("Connected to Qdrant")
    
    # ========================================
    # CV COLLECTION INDEXES
    # ========================================
    print("\n[CV] Creating indexes for cv_embeddings...")
    
    cv_indexes = {
        # Integer fields
        "cvId": PayloadSchemaType.INTEGER,
        "positionId": PayloadSchemaType.INTEGER,
        # Phase 3: array field — indexed as INTEGER so MatchAny filter is efficient
        "applied_position_ids": PayloadSchemaType.INTEGER,

        # Keyword fields (exact match) — UUID strings
        "candidateId": PayloadSchemaType.KEYWORD,
        "hrId": PayloadSchemaType.KEYWORD,
        "section": PayloadSchemaType.KEYWORD,
        "seniorityLevel": PayloadSchemaType.KEYWORD,
        "cvStatus": PayloadSchemaType.KEYWORD,
        "sourceType": PayloadSchemaType.KEYWORD,
        "skills": PayloadSchemaType.KEYWORD,

        # Boolean fields
        "is_latest": PayloadSchemaType.BOOL,

        # Integer fields
        "chunkIndex": PayloadSchemaType.INTEGER,
        "version": PayloadSchemaType.INTEGER,
        "experienceYears": PayloadSchemaType.INTEGER,
    }
    
    for field_name, field_type in cv_indexes.items():
        try:
            client.create_payload_index(
                collection_name=settings.CV_COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_type
            )
            print(f"  [OK] Created index: {field_name} ({field_type})")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  [--] Index exists: {field_name}")
            else:
                print(f"  [ERR] Error creating {field_name}: {e}")
    
    # Text index for full-text search on chunkText
    try:
        client.create_payload_index(
            collection_name=settings.CV_COLLECTION_NAME,
            field_name="chunkText",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20
            )
        )
        print(f"  [OK] Created text index: chunkText")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  [--] Text index exists: chunkText")
        else:
            print(f"  [ERR] Error creating text index: {e}")
    
    # ========================================
    # JD COLLECTION INDEXES
    # ========================================
    print("\n[JD] Creating indexes for jd_embeddings...")
    
    jd_indexes = {
        # Integer fields
        "jdId": PayloadSchemaType.INTEGER,
        "positionId": PayloadSchemaType.INTEGER,
        "version": PayloadSchemaType.INTEGER,
        "textLength": PayloadSchemaType.INTEGER,

        # Keyword fields
        "hrId": PayloadSchemaType.KEYWORD,
        # positionTitle / seniority used by chatbot-service scoring context builder
        "positionTitle": PayloadSchemaType.KEYWORD,
        "seniority": PayloadSchemaType.KEYWORD,

        # Boolean fields
        "is_latest": PayloadSchemaType.BOOL,
    }
    
    for field_name, field_type in jd_indexes.items():
        try:
            client.create_payload_index(
                collection_name=settings.JD_COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_type
            )
            print(f"  [OK] Created index: {field_name} ({field_type})")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  [--] Index exists: {field_name}")
            else:
                print(f"  [ERR] Error creating {field_name}: {e}")
    
    # Text index for JD text
    try:
        client.create_payload_index(
            collection_name=settings.JD_COLLECTION_NAME,
            field_name="jdText",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20
            )
        )
        print(f"  [OK] Created text index: jdText")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  [--] Text index exists: jdText")
        else:
            print(f"  [ERR] Error creating text index: {e}")
    
    print("\n[DONE] All indexes created successfully!")
    
    # Verify
    print("\n[INFO] Verifying collections...")
    cv_info = client.get_collection(settings.CV_COLLECTION_NAME)
    jd_info = client.get_collection(settings.JD_COLLECTION_NAME)
    
    print(f"\nCV Collection:")
    print(f"  Points: {cv_info.points_count}")
    print(f"  Indexed fields: {len(cv_info.payload_schema)}")
    
    print(f"\nJD Collection:")
    print(f"  Points: {jd_info.points_count}")
    print(f"  Indexed fields: {len(jd_info.payload_schema)}")


if __name__ == "__main__":
    create_indexes()