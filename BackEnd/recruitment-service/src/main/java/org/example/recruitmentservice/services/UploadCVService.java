package org.example.recruitmentservice.services;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.exception.CustomException;
import org.example.recruitmentservice.config.RabbitMQConfig;
import org.example.recruitmentservice.dto.request.CVUploadEvent;
import org.example.recruitmentservice.dto.response.DriveFileInfo;
import org.example.recruitmentservice.models.entity.ProcessingBatch;
import org.example.recruitmentservice.models.enums.BatchType;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.models.enums.SourceType;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
public class UploadCVService {
    private final RabbitTemplate rabbitTemplate;
    private final CandidateCVRepository candidateCVRepository;
    private final StorageService storageService;
    private final PositionRepository positionRepository;
    private final ProcessingBatchService processingBatchService;

    /**
     * Upload CV cho HR (multiple CVs cho một position cụ thể)
     */
    public ApiResponse<Map<String, Object>> uploadCVsByHR(
            List<MultipartFile> files,
            Integer positionId,
            HttpServletRequest request) {

        String role = extractAndValidateRole(request);
        if (!"HR".equalsIgnoreCase(role)) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION);
        }

        String userId = extractUserId(request);

        Positions position = positionRepository.findById(positionId)
                .orElseThrow(() -> new CustomException(ErrorCode.POSITION_NOT_FOUND));

        if (files == null || files.isEmpty()) {
            throw new CustomException(ErrorCode.FILE_NOT_FOUND);
        }

        String batchId = generateHRBatchId(positionId);

        ProcessingBatch batch = processingBatchService.createBatch(
                batchId,
                positionId,
                files.size(),
                BatchType.CV_UPLOAD);

        AtomicInteger successCounter = new AtomicInteger();
        List<CompletableFuture<Void>> futures = files.stream()
                .map(file -> CompletableFuture.runAsync(() -> {
                    try {
                        uploadSingleCV(file, position, batchId, SourceType.INTERNAL, userId);
                        successCounter.incrementAndGet();
                    } catch (Exception e) {
                        System.err.println("File failed: " + e.getMessage());
                    }
                }))
                .toList();

        java.util.concurrent.CompletableFuture.allOf(futures.toArray(new java.util.concurrent.CompletableFuture[0]))
                .join();
        int successCount = successCounter.get();

        Map<String, Object> response = new HashMap<>();
        response.put("batchId", batchId);
        response.put("message", "Please wait a moment. Your CVs are being processed.");
        response.put("totalCv", files.size());
        response.put("successCount", successCount);
        response.put("status", batch.getStatus());

        return new ApiResponse<>(200, "Batch created successfully", response);
    }

    /**
     * Upload CV cho CANDIDATE (single CV, không cần positionId).
     * Nếu Master CV đã tồn tại (re-upload): soft-delete Master cũ + toàn bộ
     * Application CVs,
     * sau đó tạo Master CV mới và kích hoạt pipeline re-embed.
     */
    public ApiResponse<Map<String, Object>> uploadCVByCandidate(
            MultipartFile file,
            HttpServletRequest request) {

        String role = extractAndValidateRole(request);
        if (!"CANDIDATE".equalsIgnoreCase(role)) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION);
        }

        String candidateId = extractUserId(request);

        if (file == null || file.isEmpty()) {
            throw new CustomException(ErrorCode.FILE_NOT_FOUND);
        }

        // Kiểm tra Master CV cũ (positionId IS NULL) — không check trên Application CVs
        Optional<CandidateCV> existingMaster = candidateCVRepository.findMasterCvByCandidateId(candidateId);
        if (existingMaster.isPresent()) {
            softDeleteMasterAndApplications(existingMaster.get());
        }

        String batchId = generateCandidateBatchId();
        ProcessingBatch batch = processingBatchService.createBatch(
                batchId,
                null,
                1,
                BatchType.CV_UPLOAD);

        try {
            CandidateCV cv = uploadSingleCV(file, null, batchId, SourceType.EXTERNAL, candidateId);

            Map<String, Object> response = new HashMap<>();
            response.put("cvId", cv.getId());
            response.put("batchId", batchId);
            response.put("message", "Your CV has been uploaded successfully and is being processed.");
            response.put("status", batch.getStatus());

            return new ApiResponse<>(200, "CV uploaded successfully", response);
        } catch (Exception e) {
            System.err.println("Candidate CV upload failed: " + e.getMessage());
            throw new CustomException(ErrorCode.FAILED_SAVE_FILE);
        }
    }

    /**
     * Soft-delete Master CV và tất cả Application CVs liên kết khi Candidate
     * re-upload.
     * Application CVs cũ được xác định qua parentCvId — Candidate phải nộp lại ứng
     * tuyển nếu muốn tiếp tục.
     */
    private void softDeleteMasterAndApplications(CandidateCV master) {
        LocalDateTime now = LocalDateTime.now();

        List<CandidateCV> applicationCVs = candidateCVRepository.findApplicationsByParentCvId(master.getId());
        applicationCVs.forEach(app -> app.setDeletedAt(now));
        candidateCVRepository.saveAll(applicationCVs);

        master.setDeletedAt(now);
        candidateCVRepository.save(master);

        System.out.println("Re-upload: soft-deleted master CV id=" + master.getId()
                + " and " + applicationCVs.size() + " application CVs");
    }

    /**
     * Core method để upload một CV
     * 
     * @param position - có thể null nếu là CANDIDATE upload
     */
    private CandidateCV uploadSingleCV(
            MultipartFile file,
            Positions position,
            String batchId,
            SourceType sourceType,
            String userId) {

        try {
            if (file == null || file.isEmpty()) {
                throw new CustomException(ErrorCode.FILE_NOT_FOUND);
            }

            // Build folder path
            String folderPath = buildFolderPath(position);

            // Upload lên Drive
            DriveFileInfo driveFileInfo = storageService.uploadCV(file, folderPath);

            // Save CV entity
            CandidateCV cv = new CandidateCV();
            cv.setPosition(position);
            cv.setDriveFileId(driveFileInfo.getFileId());
            cv.setDriveFileUrl(driveFileInfo.getWebViewLink());

            cv.setCvStatus(CVStatus.PENDING);
            cv.setCreatedAt(LocalDateTime.now());
            cv.setUpdatedAt(LocalDateTime.now());
            cv.setBatchId(batchId);
            cv.setSourceType(sourceType);
            if (sourceType == SourceType.INTERNAL) {
                cv.setHrId(userId);
            } else if (sourceType == SourceType.EXTERNAL) {
                cv.setCandidateId(userId);
            }

            candidateCVRepository.save(cv);

            // Publish event to RabbitMQ (chỉ parse, không upload)
            CVUploadEvent event = new CVUploadEvent(
                    cv.getId(),
                    driveFileInfo.getFileId(), // Gửi fileId thay vì path
                    position != null ? position.getId() : null,
                    batchId);

            rabbitTemplate.convertAndSend(RabbitMQConfig.CV_UPLOAD_QUEUE, event);
            System.out.println(
                    "Event published to RabbitMQ - CV ID: " + cv.getId() + " | FileId: " + driveFileInfo.getFileId());

            return cv;

        } catch (CustomException e) {
            System.err.println("CustomException: " + e.getMessage());
            throw e;
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            throw new CustomException(ErrorCode.FAILED_SAVE_FILE);
        }
    }

    /**
     * Build folder path cho CV trên Drive
     */
    private String buildFolderPath(Positions position) {
        if (position != null && position.getDriveFileId() != null) {
            StringBuilder path = new StringBuilder();
            path.append(position.getTitle());
            if (position.getSeniority() != null && !position.getSeniority().isBlank()) {
                path.append("/").append(position.getSeniority().trim());
            }
            path.append("/CV");
            return path.toString();
        } else {
            // CANDIDATE upload - lưu vào thư mục chung
            return "candidate-cvs/" + LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy/MM"));
        }
    }

    /**
     * Generate batch ID cho HR (có positionId)
     */
    private String generateHRBatchId(Integer positionId) {
        return "POS" + positionId + "_"
                + LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"))
                + "_B" + UUID.randomUUID().toString().substring(0, 4);
    }

    /**
     * Generate batch ID cho CANDIDATE (không có positionId)
     */
    private String generateCandidateBatchId() {
        return "CAND_"
                + LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"))
                + "_" + UUID.randomUUID().toString().substring(0, 8);
    }

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

    /**
     * Extract và validate role từ request
     */
    private String extractAndValidateRole(HttpServletRequest request) {
        String role = request.getHeader("X-User-Role");
        if (role == null) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION);
        }
        return role;
    }
}
