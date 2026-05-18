package org.example.authservice.models;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "otp_tokens", indexes = {
        @Index(name = "idx_otp_email_purpose", columnList = "email, purpose")
})
public class OtpToken {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String email;

    @Column(nullable = false)
    @Enumerated(EnumType.STRING)
    private OtpPurpose purpose;

    @Column(nullable = false)
    private String otpHash;

    @Column(nullable = false)
    private int attempts;

    @Column(nullable = false)
    private LocalDateTime expiresAt;

    @Column(nullable = false)
    private boolean used;

    /**
     * Issued after RESET_PASSWORD OTP is verified. Used to authorize the /reset-password call.
     * NULL until OTP is successfully verified.
     */
    @Column(unique = true)
    private String resetToken;

    @Column
    private LocalDateTime resetTokenExpiresAt;

    @Column(nullable = false)
    private LocalDateTime createdAt;
}
