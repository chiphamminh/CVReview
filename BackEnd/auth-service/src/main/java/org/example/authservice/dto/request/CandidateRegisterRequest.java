package org.example.authservice.dto.request;

import lombok.Data;

@Data
public class CandidateRegisterRequest {
    private String email;
    private String name;
    private String password;
}
