package org.example.recruitmentservice.services;

import org.springframework.transaction.annotation.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.exception.CustomException;
import org.example.recruitmentservice.dto.response.BatchStatusResponse;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.entity.ProcessingBatch;
import org.example.recruitmentservice.models.enums.BatchStatus;
import org.example.recruitmentservice.models.enums.BatchType;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.ProcessingBatchRepository;
import org.example.recruitmentservice.sse.SseEmitterRegistry;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;
import org.example.recruitmentservice.repository.PositionRepository;
import org.example.recruitmentservice.models.enums.JDStatus;

@Slf4j
@Service
@RequiredArgsConstructor
public class ProcessingBatchService {
    private final ProcessingBatchRepository batchRepository;
    private final CandidateCVRepository candidateCVRepository;
    private final PositionRepository positionRepository;
    private final SseEmitterRegistry sseEmitterRegistry;

    public ProcessingBatch createBatch(String batchId, Integer positionId, int total, BatchType type) {
        ProcessingBatch batch = new ProcessingBatch();
        batch.setBatchId(batchId);
        batch.setPositionId(positionId);
        batch.setTotal(total);
        batch.setSuccess(0);
        batch.setFailed(0);
        batch.setStatus(BatchStatus.PROCESSING);
        batch.setType(type);
        batch.setCreatedAt(LocalDateTime.now());

        return batchRepository.save(batch);
    }

    @Transactional
    public void incrementProcessed(String batchId, boolean isSuccess) {
        ProcessingBatch batch = batchRepository.findByBatchId(batchId)
                .orElseThrow(() -> new CustomException(ErrorCode.BATCH_NOT_FOUND));

        long actualSuccess = 0;
        long actualFailed = 0;

        if (batch.getType() == BatchType.JD_UPLOAD) {
            actualSuccess = positionRepository.countByBatchIdAndStatus(batchId, JDStatus.EMBEDDED);
            actualFailed = positionRepository.countByBatchIdAndStatus(batchId, JDStatus.FAILED);
        } else {
            actualSuccess = candidateCVRepository.countByBatchIdAndCvStatus(batchId, CVStatus.EMBEDDED);
            actualFailed = candidateCVRepository.countByBatchIdAndCvStatus(batchId, CVStatus.FAILED);
        }

        batch.setSuccess((int) actualSuccess);
        batch.setFailed((int) actualFailed);

        boolean isCompleted = batch.getProcessed() >= batch.getTotal();
        if (isCompleted) {
            batch.setStatus(BatchStatus.COMPLETED);
            batch.setCompletedAt(LocalDateTime.now());
            log.info("Batch {} completed: {}/{} processed, {} success, {} failed",
                    batchId, batch.getProcessed(), batch.getTotal(),
                    batch.getSuccess(), batch.getFailed());
        }

        batchRepository.save(batch);

        BatchStatusResponse snapshot = buildStatusSnapshot(batch, batchId);
        sseEmitterRegistry.send(batchId, snapshot);
        if (isCompleted) {
            sseEmitterRegistry.complete(batchId);
        }
    }

    public ApiResponse<BatchStatusResponse> getBatchStatus(String batchId) {
        ProcessingBatch batch = batchRepository.findByBatchId(batchId)
                .orElseThrow(() -> new CustomException(ErrorCode.BATCH_NOT_FOUND));

        BatchStatusResponse response = buildStatusSnapshot(batch, batchId);

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Batch status retrieved successfully",
                response);
    }

    /**
     * Builds a BatchStatusResponse from the given batch entity.
     * For CV_UPLOAD batches, fetches failed CV IDs from DB.
     * For JD_UPLOAD batches, failedIds is omitted (null) — not applicable.
     */
    private BatchStatusResponse buildStatusSnapshot(ProcessingBatch batch, String batchId) {
        List<Integer> failedIds = null;
        if (batch.getType() == BatchType.CV_UPLOAD) {
            failedIds = candidateCVRepository.findByBatchIdAndCvStatus(batchId, CVStatus.FAILED)
                    .stream()
                    .map(CandidateCV::getId)
                    .collect(Collectors.toList());
        }

        return BatchStatusResponse.builder()
                .batchId(batch.getBatchId())
                .processed(batch.getProcessed())
                .total(batch.getTotal())
                .success(batch.getSuccess())
                .failed(batch.getFailed())
                .failedIds(failedIds)
                .progress(BigDecimal.valueOf(batch.getProgress())
                        .setScale(2, RoundingMode.HALF_UP)
                        .doubleValue())
                .pending(batch.getPending())
                .status(batch.getStatus().name())
                .createdAt(batch.getCreatedAt())
                .completedAt(batch.getCompletedAt())
                .build();
    }
}
