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
public class ScoreTrendResponse {
    private List<WeekPoint> points;
    private Integer positionId;
    private int days;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class WeekPoint {
        private String weekLabel;
        private double avgScore;
        private long cvCount;
    }
}
