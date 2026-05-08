package org.example.recruitmentservice.listener;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.exception.CustomException;
import org.example.recruitmentservice.config.RabbitMQConfig;
import org.example.recruitmentservice.dto.request.CVChunkedEvent;
import org.example.recruitmentservice.dto.request.CVUploadEvent;
import org.example.recruitmentservice.dto.request.ChunkPayload;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.services.chunking.ChunkingService;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.example.recruitmentservice.utils.PositionUtils;

import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
public class ExtractCVListener {

    private final CandidateCVRepository candidateCVRepository;
    private final ChunkingService chunkingService;
    private final RabbitTemplate rabbitTemplate;

    @RabbitListener(queues = RabbitMQConfig.CV_EXTRACT_QUEUE, containerFactory = "cvExtractionContainerFactory")
    @Transactional
    public void handleExtract(@Payload CVUploadEvent event) {
        int cvId = event.getCvId();
        log.info("[EXTRACT] Received extraction trigger for cvId={}", cvId);

        CandidateCV cv = candidateCVRepository.findById(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        // Guard: skip CVs that already failed or were re-queued stale
        if (cv.getCvStatus() == CVStatus.FAILED || cv.getCvStatus() == CVStatus.EMBEDDED) {
            log.warn("[EXTRACT] CV {} in terminal state {}, discarding stale message", cvId, cv.getCvStatus());
            return;
        }

        if (cv.getCvContent() == null || cv.getCvContent().isBlank()) {
            log.error("[EXTRACT] CV {} has no parsed content, cannot chunk", cvId);
            markAsFailed(cv, "CV content is empty — cannot chunk and extract metadata.");
            throw new RuntimeException("CV content missing for cvId=" + cvId);
        }

        try {
            cv.setCvStatus(CVStatus.EMBEDDING);
            cv.setUpdatedAt(LocalDateTime.now());
            candidateCVRepository.save(cv);

            List<ChunkPayload> chunks = chunkingService.chunk(cv, cv.getCvContent());
            if (chunks == null || chunks.isEmpty()) {
                markAsFailed(cv, "Chunking service returned empty chunks.");
                throw new RuntimeException("Chunking failed for cvId=" + cvId);
            }

            int totalTokens = chunks.stream().mapToInt(ChunkPayload::getTokensEstimate).sum();
            String formattedTitle = PositionUtils.formatPositionTitle(cv.getPosition().getTitle(),
                    cv.getPosition().getSeniority());

            CVChunkedEvent chunkedEvent = new CVChunkedEvent(
                    cvId,
                    cv.getCandidateId(),
                    cv.getHrId(),
                    formattedTitle,
                    chunks,
                    chunks.size(),
                    totalTokens,
                    event.getBatchId());

            rabbitTemplate.convertAndSend(RabbitMQConfig.CV_EMBED_QUEUE, chunkedEvent);
            log.info("[EXTRACT] Published CVChunkedEvent for cvId={} to cv.embed.queue with {} chunks", cvId,
                    chunks.size());

        } catch (CustomException e) {
            throw e;
        } catch (Exception e) {
            log.error("[EXTRACT] Extraction/Chunking failed for cvId={}: {}", cvId, e.getMessage(), e);
            // Re-throw để RabbitMQ route sang cv.extract.queue.dlq
            throw new RuntimeException("Extraction/Chunking failed for cvId=" + cvId, e);
        }
    }

    private void markAsFailed(CandidateCV cv, String reason) {
        cv.setCvStatus(CVStatus.FAILED);
        cv.setErrorMessage(reason);
        cv.setFailedAt(LocalDateTime.now());
        cv.setUpdatedAt(LocalDateTime.now());
        candidateCVRepository.save(cv);
    }
}
