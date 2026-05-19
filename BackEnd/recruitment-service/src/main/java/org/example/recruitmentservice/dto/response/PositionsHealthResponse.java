package org.example.recruitmentservice.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PositionsHealthResponse {
    private List<PositionHealth> positions;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class PositionHealth {
        private int id;
        private String title;
        private String seniority;
        private double minimumFitScore;
        private long poolSize;
        private double avgScore;
        private long qualifiedCount;
        /** HEALTHY (≥10 qualified) | MEDIUM (4-9) | LOW (≤3) */
        private String healthStatus;
    }
}
