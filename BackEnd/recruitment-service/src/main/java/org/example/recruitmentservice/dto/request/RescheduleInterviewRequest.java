package org.example.recruitmentservice.dto.request;

import lombok.*;
import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class RescheduleInterviewRequest {
    private LocalDateTime interviewDate;
    private String customMessage;
}
