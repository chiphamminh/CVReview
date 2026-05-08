package org.example.recruitmentservice.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Full JD text payload returned by the internal chatbot API.
 * Used by chatbot-service for Small-to-Big retrieval:
 * Qdrant finds chunk → extract positionId → call this API → feed full JD to the LLM.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PositionDetailsResponse {
    private Integer id;
    private String title;
    private String seniority;
    private String jdText;
}
