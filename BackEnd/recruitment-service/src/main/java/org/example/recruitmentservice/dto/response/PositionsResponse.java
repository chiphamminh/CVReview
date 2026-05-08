package org.example.recruitmentservice.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@Builder
public class PositionsResponse {
    private Integer id;
    private String hrId;
    private String title;
    private String seniority;
    private List<String> skills;
    private Double minimumFitScore;
    private String driveFileUrl;
    private String jdText;
    private Boolean isActive;
    private LocalDateTime openedAt;
    private org.example.recruitmentservice.models.enums.JDStatus status;
    private String batchId;
    private Integer totalCVs;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

}
