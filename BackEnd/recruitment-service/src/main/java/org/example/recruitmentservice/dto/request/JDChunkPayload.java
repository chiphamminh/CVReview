package org.example.recruitmentservice.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a single semantic chunk sliced from a Job Description Markdown document.
 * Consumed by the embedding-service to create individual Qdrant points per chunk
 * while preserving the parent positionId for Small-to-Big retrieval.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class JDChunkPayload {
    private Integer positionId;
    private String positionTitle;
    private String seniority;
    private int chunkIndex;
    private String sectionName;
    private String chunkText;
    private int words;
    private int tokensEstimate;
}
