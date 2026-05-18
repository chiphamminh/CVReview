package org.example.authservice.dto.response;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class VerifyResetOtpResponse {
    private String resetToken;
}
