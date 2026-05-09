package org.example.recruitmentservice.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.enums.MatchStatus;
import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.models.enums.SourceType;

import java.time.LocalDateTime;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@Builder
public class CandidateCVResponse {
    private int cvId;
    private int positionId;
    private String positionTitle;
    private String email;
    private String name;
    private String batchId;
    private String driveFileUrl;
    private Integer technicalScore;
    private Integer experienceScore;
    private MatchStatus overallStatus;
    private String aiAssessment;
    private String learningPath;
    private CVStatus status;
    private SourceType sourceType;
    private RecruitmentStage recruitmentStage;
    private String errorMessage;
    private LocalDateTime failedAt;
    private Integer retryCount;
    private Boolean canRetry;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime appliedDate;
    private LocalDateTime interviewSchedule;
}
