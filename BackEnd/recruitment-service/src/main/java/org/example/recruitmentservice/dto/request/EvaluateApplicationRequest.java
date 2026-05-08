package org.example.recruitmentservice.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.example.recruitmentservice.models.enums.MatchStatus;

@Getter
@NoArgsConstructor
public class EvaluateApplicationRequest {
    private Integer appCvId;
    private Integer positionId;
    private Integer technicalScore;
    private Integer experienceScore;
    private MatchStatus overallStatus;
    private String aiAssessment;
    private String learningPath;
    private String sessionId;
}
