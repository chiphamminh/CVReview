package org.example.authservice.repository;

import org.example.authservice.models.OtpPurpose;
import org.example.authservice.models.OtpToken;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;

public interface OtpTokenRepository extends JpaRepository<OtpToken, Long> {

    Optional<OtpToken> findTopByEmailAndPurposeAndUsedFalseOrderByCreatedAtDesc(
            String email, OtpPurpose purpose);

    Optional<OtpToken> findByResetToken(String resetToken);

    @Modifying
    @Transactional
    @Query("DELETE FROM OtpToken o WHERE o.expiresAt < :now AND o.used = false")
    void deleteExpiredUnusedOtps(@Param("now") LocalDateTime now);

    @Modifying
    @Transactional
    @Query("DELETE FROM OtpToken o WHERE o.resetTokenExpiresAt < :now AND o.used = true")
    void deleteExpiredResetTokens(@Param("now") LocalDateTime now);
}
