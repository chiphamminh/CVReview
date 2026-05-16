package org.example.recruitmentservice.dto.response;

import lombok.*;

import java.time.LocalDateTime;
import java.util.List;

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
    private List<FailedCVItem> cvs;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FailedCVItem {
        private int cvId;
        private String fileName;
        private String name;
        private String email;
        private String driveFileUrl;
        private String errorMessage;
    }
}
