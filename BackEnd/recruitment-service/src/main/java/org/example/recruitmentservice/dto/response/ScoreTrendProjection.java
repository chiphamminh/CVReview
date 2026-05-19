package org.example.recruitmentservice.dto.response;

public interface ScoreTrendProjection {
    Integer getYr();
    Integer getWk();
    Double getAvgScore();
    Long getCvCount();
}
