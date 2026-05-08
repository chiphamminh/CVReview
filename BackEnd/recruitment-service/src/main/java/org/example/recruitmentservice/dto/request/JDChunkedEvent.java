package org.example.recruitmentservice.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * RabbitMQ event published after a JD has been parsed and chunked.
 * Replaces the obsolete JDParsedEvent — now carries structured chunk payloads
 * instead of a raw full-text string, enabling Small-to-Big RAG on the consumer side.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class JDChunkedEvent {
    private Integer positionId;
    private String positionTitle;
    private String seniority;
    private List<JDChunkPayload> chunks;
    private Integer totalChunks;
    private Integer totalTokens;
    private String batchId;
}
