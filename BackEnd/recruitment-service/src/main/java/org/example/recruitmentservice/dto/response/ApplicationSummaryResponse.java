package org.example.recruitmentservice.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Getter;

/**
 * DTO trả về thông tin CV đã được liên kết với một position.
 * Bao gồm cả HR-uploaded CVs (candidateId = null) và Candidate-applied CVs.
 * Python hr_graph dùng sourceType để phân biệt 2 mode và build sql_metadata map đúng.
 */
@Getter
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApplicationSummaryResponse {
    private String candidateId;   // null nếu là HR-uploaded CV
    private String candidateName;
    private String candidateEmail;
    private Integer appCvId;      // cvId trong Qdrant (HR) hoặc id của applied CV (CANDIDATE)
    private Integer masterCvId;   // cvId trong Qdrant (CANDIDATE) — cầu nối để map name
    private String sourceType;    // "INTERNAL" | "EXTERNAL"
    private Integer score;
    private String aiAssessment;
}
