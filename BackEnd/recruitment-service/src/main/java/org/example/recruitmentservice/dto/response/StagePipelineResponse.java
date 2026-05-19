package org.example.recruitmentservice.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StagePipelineResponse {
    private long applied;
    private long interviewScheduled;
    private long interviewed;
    private long offer;
    private long accepted;
    private long rejected;
    private Integer positionId;
}
