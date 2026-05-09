package org.example.recruitmentservice.dto.response;

import lombok.*;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BatchStatusResponse {
    private String batchId;
    private Integer processed;
    private Integer total;
    private Integer success;
    private Integer failed;
    /** Populated only for CV_UPLOAD batches; null/absent for JD_UPLOAD. */
    private List<Integer> failedIds;
    private Double progress;
    private Integer pending;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime completedAt;
}
