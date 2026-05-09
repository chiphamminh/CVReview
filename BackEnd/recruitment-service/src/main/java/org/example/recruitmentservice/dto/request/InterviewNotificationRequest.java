package org.example.recruitmentservice.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.AllArgsConstructor;
import lombok.Builder;

/**
 * Payload gửi email phỏng vấn từ HR chatbot.
 * emailType: INTERVIEW_INVITE | OFFER_LETTER
 * interviewDate: ISO format string, nullable (không cần cho OFFER_LETTER)
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InterviewNotificationRequest {
    private Integer appCvId;
    private String candidateId;
    private String candidateEmail;
    private String candidateName;
    private Integer positionId;
    private String positionName;
    private String emailType;
    private String interviewDate;
    private String customMessage;
    private String sessionId;
    private String benefit;
    private String salary;
    private String startDate;
    private String offerExpirationDate;
    private String additionalNote;
}
