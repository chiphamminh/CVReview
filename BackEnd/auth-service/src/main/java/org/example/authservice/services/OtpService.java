package org.example.authservice.services;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.authservice.models.OtpPurpose;
import org.example.authservice.models.OtpToken;
import org.example.authservice.repository.OtpTokenRepository;
import org.example.commonlibrary.exception.CustomException;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.security.SecureRandom;
import java.time.LocalDateTime;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class OtpService {

    private final OtpTokenRepository otpTokenRepository;
    private final EmailService emailService;
    private final PasswordEncoder passwordEncoder;

    @Value("${otp.expiration-minutes:5}")
    private int otpExpirationMinutes;

    @Value("${otp.max-attempts:5}")
    private int maxAttempts;

    @Value("${otp.reset-token-expiration-minutes:10}")
    private int resetTokenExpirationMinutes;

    private static final SecureRandom RANDOM = new SecureRandom();

    /**
     * Generates a new OTP for the given email + purpose.
     * Invalidates any previous pending OTP for the same email + purpose before creating a new one.
     */
    @Transactional
    public void generateAndSend(String email, OtpPurpose purpose) {
        // Invalidate previous pending OTP if any
        otpTokenRepository
                .findTopByEmailAndPurposeAndUsedFalseOrderByCreatedAtDesc(email, purpose)
                .ifPresent(old -> {
                    old.setUsed(true);
                    otpTokenRepository.save(old);
                });

        String rawOtp = generateSixDigitOtp();
        String hashedOtp = passwordEncoder.encode(rawOtp);

        OtpToken token = OtpToken.builder()
                .email(email.toLowerCase())
                .purpose(purpose)
                .otpHash(hashedOtp)
                .attempts(0)
                .used(false)
                .expiresAt(LocalDateTime.now().plusMinutes(otpExpirationMinutes))
                .createdAt(LocalDateTime.now())
                .build();

        otpTokenRepository.save(token);
        emailService.sendOtpEmail(email, rawOtp, purpose.name());
        log.info("[OTP] Generated and sent for email={} purpose={}", email, purpose);
    }

    /**
     * Verifies OTP. On success marks it as used.
     * For RESET_PASSWORD purpose, also issues a resetToken and returns it.
     * For REGISTRATION purpose, returns null (caller just checks no exception).
     */
    @Transactional
    public String verifyOtp(String email, String rawOtp, OtpPurpose purpose) {
        OtpToken token = otpTokenRepository
                .findTopByEmailAndPurposeAndUsedFalseOrderByCreatedAtDesc(email.toLowerCase(), purpose)
                .orElseThrow(() -> new CustomException(ErrorCode.OTP_INVALID));

        if (token.getExpiresAt().isBefore(LocalDateTime.now())) {
            token.setUsed(true);
            otpTokenRepository.save(token);
            throw new CustomException(ErrorCode.OTP_EXPIRED);
        }

        if (token.getAttempts() >= maxAttempts) {
            token.setUsed(true);
            otpTokenRepository.save(token);
            throw new CustomException(ErrorCode.OTP_MAX_ATTEMPTS);
        }

        if (!passwordEncoder.matches(rawOtp, token.getOtpHash())) {
            token.setAttempts(token.getAttempts() + 1);
            if (token.getAttempts() >= maxAttempts) {
                token.setUsed(true);
            }
            otpTokenRepository.save(token);
            throw new CustomException(ErrorCode.OTP_INVALID);
        }

        token.setUsed(true);

        if (purpose == OtpPurpose.RESET_PASSWORD) {
            String resetToken = UUID.randomUUID().toString();
            token.setResetToken(resetToken);
            token.setResetTokenExpiresAt(LocalDateTime.now().plusMinutes(resetTokenExpirationMinutes));
            otpTokenRepository.save(token);
            log.info("[OTP] RESET_PASSWORD OTP verified for email={}, resetToken issued", email);
            return resetToken;
        }

        otpTokenRepository.save(token);
        log.info("[OTP] REGISTRATION OTP verified for email={}", email);
        return null;
    }

    /**
     * Validates a resetToken and returns the associated email.
     * Deletes the token after use to prevent reuse.
     */
    @Transactional
    public String consumeResetToken(String resetToken) {
        OtpToken token = otpTokenRepository.findByResetToken(resetToken)
                .orElseThrow(() -> new CustomException(ErrorCode.RESET_TOKEN_INVALID));

        if (token.getResetTokenExpiresAt() == null
                || token.getResetTokenExpiresAt().isBefore(LocalDateTime.now())) {
            otpTokenRepository.delete(token);
            throw new CustomException(ErrorCode.RESET_TOKEN_INVALID);
        }

        String email = token.getEmail();
        otpTokenRepository.delete(token);
        return email;
    }

    @Scheduled(cron = "0 0 3 * * *", zone = "Asia/Ho_Chi_Minh")
    @Transactional
    public void cleanupExpiredOtps() {
        LocalDateTime now = LocalDateTime.now();
        otpTokenRepository.deleteExpiredUnusedOtps(now);
        otpTokenRepository.deleteExpiredResetTokens(now);
        log.info("[OTP] Scheduled cleanup of expired OTPs done");
    }

    private String generateSixDigitOtp() {
        int otp = 100000 + RANDOM.nextInt(900000);
        return String.valueOf(otp);
    }
}
