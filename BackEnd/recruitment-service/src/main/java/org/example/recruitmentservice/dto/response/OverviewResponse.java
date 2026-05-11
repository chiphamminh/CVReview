package org.example.recruitmentservice.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OverviewResponse {
    private long totalCvsScored;
    private long totalCvsPassed;
    private double avgMatchingScore;
    private double successMatchRate;
    private int days;
}
