package org.example.authservice.dto.request;

import lombok.Data;

@Data
public class VerifyRegisterRequest {
    private String email;
    private String otp;
    private String name;
    private String password;
}
