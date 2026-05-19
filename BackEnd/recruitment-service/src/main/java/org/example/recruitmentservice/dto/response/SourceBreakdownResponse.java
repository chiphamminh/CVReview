package org.example.recruitmentservice.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SourceBreakdownResponse {
    private long internalCount;
    private long internalScored;
    private long internalPassed;
    private double internalAvgScore;
    private long externalCount;
    private long externalScored;
    private long externalPassed;
    private double externalAvgScore;
    private int days;
    private Integer positionId;
}
