package org.example.authservice.dto.request;

import lombok.Data;

@Data
public class CandidateLoginRequest {
    private String email;
    private String password;
}
