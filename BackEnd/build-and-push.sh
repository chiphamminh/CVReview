#!/bin/bash

# ==============================================================================
# Build & Push Script cho Microservices
# ==============================================================================
# Mục đích: Build tất cả services và push lên Docker Hub
# Sử dụng: ./build-and-push.sh v1.0.0
# ==============================================================================

set -e  # Dừng script nếu có lỗi

# Kiểm tra version argument
if [ -z "$1" ]; then
    echo "Error: Version không được để trống!"
    echo "Sử dụng: ./build-and-push.sh v1.0.0"
    exit 1
fi

VERSION=$1
DOCKER_USERNAME="toilachi1604"  # Thay bằng username Docker Hub của bạn

echo "Bắt đầu build và push images với version: $VERSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Đăng nhập Docker Hub (chỉ cần làm 1 lần)
echo "Đăng nhập Docker Hub..."
docker login

# Danh sách services cần build
SERVICES=(
    "auth-service"
    "recruitment-service"
    "api-gateway"
    "embedding-api"
    "chatbot-service"
)

# Build từ docker-compose với file gốc có build context
echo ""
echo "Build tất cả services..."
docker compose -f docker-compose-build.yml build

# Tag và push từng service
for SERVICE in "${SERVICES[@]}"; do
    echo ""
    echo "Processing: $SERVICE"
    echo "──────────────────────────────────────────────────────────"

    # Tag image
    LOCAL_IMAGE="${SERVICE}:latest"
    REMOTE_IMAGE="${DOCKER_USERNAME}/${SERVICE}:${VERSION}"
    REMOTE_LATEST="${DOCKER_USERNAME}/${SERVICE}:latest"

    echo "Tagging: $LOCAL_IMAGE → $REMOTE_IMAGE"
    docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"
    docker tag "$LOCAL_IMAGE" "$REMOTE_LATEST"

    # Push image
    echo "Pushing: $REMOTE_IMAGE"
    docker push "$REMOTE_IMAGE"

    echo "Pushing: $REMOTE_LATEST"
    docker push "$REMOTE_LATEST"

    echo "  Done: $SERVICE"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Hoàn thành! Tất cả images đã được push lên Docker Hub"
echo ""
echo "Summary:"
echo "   Version: $VERSION"
echo "   Services: ${#SERVICES[@]} services"
echo ""
echo "Images trên Docker Hub:"
for SERVICE in "${SERVICES[@]}"; do
    echo "   • https://hub.docker.com/r/${DOCKER_USERNAME}/${SERVICE}"
done
echo ""
echo "Next steps:"
echo "   1. Cập nhật version trong docker-compose.yml thành: $VERSION"
echo "   2. Gửi docker-compose.yml + folder env/ cho đồng nghiệp"
echo "   3. Đồng nghiệp chạy: docker compose pull && docker compose up -d"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"