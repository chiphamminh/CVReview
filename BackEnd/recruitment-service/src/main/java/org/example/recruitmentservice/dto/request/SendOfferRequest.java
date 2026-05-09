package org.example.recruitmentservice.dto.request;

import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SendOfferRequest {
    private String benefit;
    private String salary;
    private String startDate;
    private String offerExpirationDate;
    private String additionalNote;
}
