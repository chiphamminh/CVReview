package org.example.recruitmentservice.dto.response;

import lombok.*;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FailedBatchSummary {
    private String batchId;
    private Integer positionId;
    private String positionTitle;
    private int failedCount;
    private LocalDateTime uploadedAt;
    private LocalDateTime failedAt;
    private String errorMessage;
}
