package org.example.recruitmentservice.dto.request;

import lombok.*;
import org.example.recruitmentservice.models.enums.RecruitmentStage;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class UpdateStageRequest {
    private RecruitmentStage recruitmentStage;
}
