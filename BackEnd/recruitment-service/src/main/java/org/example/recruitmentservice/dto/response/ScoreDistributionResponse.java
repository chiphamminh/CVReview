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
public class ScoreDistributionResponse {
    private List<ScoreBucket> buckets;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class ScoreBucket {
        private String range;
        private String label;
        private long count;
    }
}
