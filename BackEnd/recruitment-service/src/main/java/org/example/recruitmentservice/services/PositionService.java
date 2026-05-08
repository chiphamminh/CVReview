package org.example.recruitmentservice.services;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.dto.response.PageResponse;
import org.example.commonlibrary.exception.CustomException;
import org.example.commonlibrary.utils.PageUtil;
import org.example.recruitmentservice.client.LlamaParseClient;
import org.example.recruitmentservice.config.RabbitMQConfig;
import org.example.recruitmentservice.dto.request.JDChunkPayload;
import org.example.recruitmentservice.dto.request.JDChunkedEvent;
import org.example.recruitmentservice.dto.request.PositionsRequest;
import org.example.recruitmentservice.dto.response.DriveFileInfo;
import org.example.recruitmentservice.dto.response.PositionsResponse;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.example.recruitmentservice.services.chunking.JDChunkingService;
import org.example.recruitmentservice.models.enums.JDStatus;
import org.example.recruitmentservice.models.enums.BatchType;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;
import org.example.recruitmentservice.utils.PositionUtils;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Slf4j
@Service
@RequiredArgsConstructor
public class PositionService {
    private final PositionRepository positionRepository;
    private final LlamaParseClient llamaParseClient;
    private final StorageService storageService;
    private final CandidateCVRepository candidateCVRepository;
    private final RabbitTemplate rabbitTemplate;
    private final RestTemplate restTemplate;
    private final JDChunkingService jdChunkingService;
    private final ProcessingBatchService processingBatchService;

    @Value("${EMBEDDING_SERVICE_URL}")
    private String embeddingServiceUrl;

    @Transactional
    public ApiResponse<PositionsResponse> createPosition(PositionsRequest positionsRequest,
            HttpServletRequest request) {
        String hrId = extractUserId(request);

        String title = positionsRequest.getTitle();
        String seniority = positionsRequest.getSeniority();
        if (title == null || title.isBlank()) {
            throw new CustomException(ErrorCode.MISSING_NAME_AND_LEVEL);
        }

        // Check duplicate by title + seniority
        Optional<Positions> existing = positionRepository.findByTitleAndSeniority(title.trim(), seniority);
        if (existing.isPresent()) {
            throw new CustomException(ErrorCode.DUPLICATE_POSITION);
        }

        DriveFileInfo driveFileInfo = storageService.uploadJD(
                positionsRequest.getFile(),
                positionsRequest.getTitle(),
                null,
                positionsRequest.getSeniority());

        String batchId = "JD_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + "_"
                + UUID.randomUUID().toString().substring(0, 4);
        processingBatchService.createBatch(batchId, null, 1, BatchType.JD_UPLOAD);

        Positions position = new Positions();
        position.setHrId(hrId);
        position.setTitle(positionsRequest.getTitle().trim());
        position.setSeniority(positionsRequest.getSeniority());
        position.setSkills(
                positionsRequest.getSkills() != null ? positionsRequest.getSkills() : new java.util.ArrayList<>());
        position.setJobDescription("Processing...");
        position.setDriveFileId(driveFileInfo.getFileId());
        position.setDriveFileUrl(driveFileInfo.getWebViewLink());
        position.setBatchId(batchId);
        position.setStatus(JDStatus.PENDING);
        position.setCreatedAt(LocalDateTime.now());
        position.setUpdatedAt(LocalDateTime.now());

        Positions positionSaved = positionRepository.save(position);

        // Async Processing
        CompletableFuture.runAsync(() -> {
            String tempFilePath = null;
            try {
                // Update to PARSING
                Positions p = positionRepository.findById(positionSaved.getId());
                if (p != null) {
                    p.setStatus(JDStatus.PARSING);
                    p.setUpdatedAt(LocalDateTime.now());
                    positionRepository.save(p);
                }

                tempFilePath = storageService.downloadFileToTemp(driveFileInfo.getFileId());
                String jdText = llamaParseClient.parseJD(tempFilePath);

                p = positionRepository.findById(positionSaved.getId());
                if (p != null) {
                    p.setStatus(JDStatus.EMBEDDING);
                    p.setJobDescription(jdText);
                    p.setUpdatedAt(LocalDateTime.now());
                    positionRepository.save(p);

                    // Chunk & Publish
                    List<JDChunkPayload> chunks = jdChunkingService.chunk(
                            p.getId(), p.getTitle(),
                            p.getSeniority(), jdText);
                    if (chunks.isEmpty()) {
                        log.warn("[Position] JD chunking produced no chunks for position {}, failing", p.getId());
                        p.setStatus(JDStatus.FAILED);
                        p.setErrorMessage("No chunks produced from JD text");
                        positionRepository.save(p);
                        processingBatchService.incrementProcessed(batchId, false);
                    } else {
                        publishJDChunkedEvent(p, chunks);
                    }
                }
            } catch (Exception e) {
                log.error("JD Parse/Chunk error details: " + e.getMessage(), e);
                Positions p = positionRepository.findById(positionSaved.getId());
                if (p != null) {
                    p.setStatus(JDStatus.FAILED);
                    p.setErrorMessage("Parsing failed: " + e.getMessage());
                    p.setUpdatedAt(LocalDateTime.now());
                    positionRepository.save(p);
                }
                processingBatchService.incrementProcessed(batchId, false);
            } finally {
                if (tempFilePath != null) {
                    storageService.deleteTempFile(tempFilePath);
                }
            }
        });

        PositionsResponse response = PositionsResponse.builder()
                .id(positionSaved.getId())
                .hrId(positionSaved.getHrId())
                .title(positionSaved.getTitle())
                .seniority(positionSaved.getSeniority())
                .skills(positionSaved.getSkills())
                .driveFileUrl(positionSaved.getDriveFileUrl())
                .status(positionSaved.getStatus())
                .batchId(positionSaved.getBatchId())
                .createdAt(positionSaved.getCreatedAt())
                .updatedAt(positionSaved.getUpdatedAt())
                .build();

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                ErrorCode.SUCCESS.getMessage(),
                response);
    }

    public ApiResponse<List<PositionsResponse>> getPositions(String title, String seniority) {
        List<Positions> positionsList = positionRepository.findByFilters(title, seniority);

        if (positionsList.isEmpty()) {
            throw new CustomException(ErrorCode.POSITION_NOT_FOUND);
        }

        List<PositionsResponse> response = positionsList.stream()
                .map(this::toResponse)
                .toList();

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                ErrorCode.SUCCESS.getMessage(),
                response);
    }

    public ApiResponse<PositionsResponse> getJdText(int positionId) {
        Positions position = positionRepository.findById(positionId);
        if (position == null) {
            throw new CustomException(ErrorCode.POSITION_NOT_FOUND);
        }

        PositionsResponse positionsResponse = PositionsResponse.builder()
                .title(position.getTitle())
                .jdText(position.getJobDescription())
                .build();

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                ErrorCode.SUCCESS.getMessage(),
                positionsResponse);
    }

    public ApiResponse<PageResponse<PositionsResponse>> getAllPositions(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<Positions> positionPage = positionRepository.findAll(pageable);

        Page<PositionsResponse> mappedPage = positionPage.map(this::toResponse);

        return ApiResponse.<PageResponse<PositionsResponse>>builder()
                .statusCode(ErrorCode.SUCCESS.getCode())
                .message("Fetched all positions successfully")
                .data(PageUtil.toPageResponse(mappedPage))
                .timestamp(LocalDateTime.now())
                .build();
    }

    public ApiResponse<List<PositionsResponse>> searchPositions(String keyword) {
        if (keyword == null || keyword.trim().isEmpty()) {
            return new ApiResponse<>(ErrorCode.POSITION_NOT_FOUND.getCode(),
                    ErrorCode.POSITION_NOT_FOUND.getMessage());
        }

        String[] words = keyword.trim().toLowerCase().split("\\s+");
        List<Positions> all = positionRepository.findAll();

        List<Positions> filtered = all.stream()
                .filter(p -> {
                    String combined = (p.getTitle() + " " + p.getSeniority() + " "
                            + String.join(" ", p.getSkills()))
                            .toLowerCase();
                    return Arrays.stream(words).allMatch(combined::contains);
                })
                .toList();

        List<PositionsResponse> responseList = filtered.stream()
                .map(this::toResponse)
                .toList();

        return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), ErrorCode.SUCCESS.getMessage(), responseList);
    }

    /**
     * Update metadata của một Position (title, seniority, skills, minimumFitScore).
     * File JD không được phép cập nhật — nếu cần đổi JD, xóa position và tạo mới.
     */
    @Transactional
    public void updatePosition(Integer positionId, PositionsRequest positionsRequest) {
        Positions position = positionRepository.findById(positionId)
                .orElseThrow(() -> new CustomException(ErrorCode.POSITION_NOT_FOUND));

        if (positionsRequest.getTitle() != null && !positionsRequest.getTitle().trim().isEmpty()) {
            position.setTitle(positionsRequest.getTitle().trim());
        }
        if (positionsRequest.getSeniority() != null && !positionsRequest.getSeniority().trim().isEmpty()) {
            position.setSeniority(positionsRequest.getSeniority().trim());
        }
        if (positionsRequest.getSkills() != null && !positionsRequest.getSkills().isEmpty()) {
            position.setSkills(positionsRequest.getSkills());
        }

        position.setUpdatedAt(LocalDateTime.now());
        positionRepository.save(position);
    }

    @Transactional
    public void deletePositions(List<Integer> positionIds) {

        for (Integer positionId : positionIds) {
            Positions position = positionRepository.findById(positionId)
                    .orElseThrow(() -> new CustomException(ErrorCode.POSITION_NOT_FOUND));

            boolean hasCV = !candidateCVRepository.findListCVsByPositionId(positionId).isEmpty();
            if (hasCV) {
                throw new CustomException(ErrorCode.CAN_NOT_DELETE_POSITION);
            }

            // Xóa embeddings trên Python service
            try {
                String url = embeddingServiceUrl + "/jd/" + positionId;
                restTemplate.delete(url);
                System.out.println("Deleted embeddings for position: " + positionId);
            } catch (Exception e) {
                System.err.println("Failed to delete embeddings for position " + positionId + ": " + e.getMessage());
            }

            // Xóa file trên Drive
            try {
                if (position.getDriveFileId() != null) {
                    storageService.deleteFile(position.getDriveFileId());
                }
            } catch (Exception e) {
                throw new CustomException(ErrorCode.FILE_DELETE_FAILED);
            }

            positionRepository.delete(position);
        }
    }

    // HELPER METHOD

    /**
     * Extract User ID từ request header
     */
    private String extractUserId(HttpServletRequest request) {
        String userId = request.getHeader("X-User-Id");
        if (userId == null || userId.isBlank()) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION);
        }
        return userId;
    }

    private PositionsResponse toResponse(Positions position) {
        int totalCVs = candidateCVRepository.countByPositionId(position.getId());

        return PositionsResponse.builder()
                .id(position.getId())
                .hrId(position.getHrId())
                .title(position.getTitle())
                .seniority(position.getSeniority())
                .skills(position.getSkills())
                .minimumFitScore(position.getMinimumFitScore())
                .driveFileUrl(position.getDriveFileUrl())
                .totalCVs(totalCVs)
                .isActive(position.isActive())
                .openedAt(position.getOpenedAt())
                .createdAt(position.getCreatedAt())
                .build();
    }

    /**
     * Publishes a {@link JDChunkedEvent} to the JD chunked exchange after
     * transaction commit.
     */
    private void publishJDChunkedEvent(Positions position, List<JDChunkPayload> chunks) {
        try {
            int totalTokens = chunks.stream().mapToInt(JDChunkPayload::getTokensEstimate).sum();

            JDChunkedEvent event = new JDChunkedEvent(
                    position.getId(),
                    PositionUtils.formatPositionTitle(position.getSeniority(), position.getTitle()),
                    position.getSeniority(),
                    chunks,
                    chunks.size(),
                    totalTokens,
                    position.getBatchId());
            rabbitTemplate.convertAndSend(
                    RabbitMQConfig.JD_CHUNKED_EXCHANGE,
                    RabbitMQConfig.JD_CHUNKED_ROUTING_KEY,
                    event);
            log.info("[Position] Published JDChunkedEvent for position {} with {} chunks ({} tokens)",
                    position.getId(), chunks.size(), totalTokens);
        } catch (Exception e) {
            log.error("[Position] Failed to publish JDChunkedEvent for position {}: {}",
                    position.getId(), e.getMessage(), e);
        }
    }
}
