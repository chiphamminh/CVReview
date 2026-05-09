package org.example.recruitmentservice.controller;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.recruitmentservice.dto.response.BatchStatusResponse;
import org.example.recruitmentservice.models.enums.BatchStatus;
import org.example.recruitmentservice.services.ProcessingBatchService;
import org.example.recruitmentservice.sse.SseEmitterRegistry;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/tracking")
@RequiredArgsConstructor
public class ProcessingBatchController {

    private final ProcessingBatchService processingBatchService;
    private final SseEmitterRegistry sseEmitterRegistry;

    /** Fallback REST polling endpoint. FE uses this to fetch current status on reconnect. */
    @PreAuthorize("hasAnyRole('HR', 'CANDIDATE')")
    @GetMapping("/{batchId}/status")
    public ResponseEntity<ApiResponse<BatchStatusResponse>> getBatchStatus(@PathVariable String batchId) {
        return ResponseEntity.ok(processingBatchService.getBatchStatus(batchId));
    }

    /**
     * SSE streaming endpoint. FE opens this once and receives push events on every item processed.
     * Timeout is 5 minutes — sufficient for the largest expected batches.
     *
     * If the batch is already COMPLETED when the client subscribes (e.g., on reconnect after
     * all items failed), the final snapshot is sent immediately and the stream is closed
     * to prevent the emitter from hanging until timeout.
     */
    @PreAuthorize("hasAnyRole('HR', 'CANDIDATE')")
    @GetMapping(value = "/{batchId}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamBatchStatus(@PathVariable String batchId) {
        SseEmitter emitter = new SseEmitter(300_000L);
        sseEmitterRegistry.register(batchId, emitter);

        try {
            BatchStatusResponse current = processingBatchService.getBatchStatus(batchId).getData();
            emitter.send(SseEmitter.event().name("batch-update").data(current));

            // If batch already finished, close the stream immediately
            if (BatchStatus.COMPLETED.name().equals(current.getStatus())) {
                emitter.send(SseEmitter.event().name("batch-completed").data("DONE"));
                emitter.complete();
            }
        } catch (Exception e) {
            emitter.completeWithError(e);
        }

        return emitter;
    }
}

