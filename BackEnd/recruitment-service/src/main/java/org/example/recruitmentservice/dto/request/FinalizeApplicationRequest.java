package org.example.recruitmentservice.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.example.recruitmentservice.models.enums.MatchStatus;

/**
 * Payload finalize_application từ chatbot-service.
 * Chứa kết quả đánh giá đa chiều từ LLM scoring node.
 */
@Getter
@NoArgsConstructor
public class FinalizeApplicationRequest {
    private String candidateId;
    private Integer positionId;
    private Integer technicalScore;
    private Integer experienceScore;
    private MatchStatus overallStatus;
    private String aiAssessment;
    private String learningPath;
    private String sessionId;
}
