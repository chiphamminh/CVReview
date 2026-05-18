package org.example.authservice.services;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailService {

    private final JavaMailSender mailSender;

    @Value("${spring.mail.username}")
    private String fromEmail;

    @Async
    public void sendOtpEmail(String toEmail, String otp, String purpose) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);

            if ("REGISTRATION".equals(purpose)) {
                message.setSubject("[CV Review] Verify your email");
                message.setText("""
                        Welcome to CV Review!

                        Your OTP verification code is: %s

                        This code expires in 5 minutes.
                        Do not share this code with anyone.
                        """.formatted(otp));
            } else {
                message.setSubject("[CV Review] Reset your password");
                message.setText("""
                        You requested a password reset on CV Review.

                        Your OTP code is: %s

                        This code expires in 5 minutes.
                        If you did not request this, please ignore this email.
                        """.formatted(otp));
            }

            mailSender.send(message);
            log.info("[EMAIL] OTP sent to {} for purpose={}", toEmail, purpose);
        } catch (Exception e) {
            log.error("[EMAIL] Failed to send OTP to {}: {}", toEmail, e.getMessage());
            throw new RuntimeException("Failed to send email", e);
        }
    }
}
