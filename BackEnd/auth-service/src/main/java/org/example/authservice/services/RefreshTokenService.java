package org.example.authservice.services;

import org.example.authservice.models.RefreshToken;
import org.example.authservice.models.Users;
import org.example.authservice.repository.RefreshTokenRepository;
import org.example.authservice.security.JwtUtil;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

@Service
public class RefreshTokenService {

    // Giới hạn số token active tối đa per user (5 thiết bị đồng thời)
    private static final int MAX_ACTIVE_TOKENS_PER_USER = 5;

    @Value("${jwt.refresh-expiration}")
    private long refreshDuration;

    @Autowired
    private RefreshTokenRepository refreshTokenRepository;

    @Autowired
    private JwtUtil jwtUtil;

    /**
     * Tạo refresh token mới cho user.
     * Sử dụng Insert-First rồi Delete-Oldest bằng 1 native query duy nhất
     * để tránh race condition và deadlock khi có nhiều request login cùng lúc.
     */
    @Transactional
    public RefreshToken createRefreshToken(Users user) {
        Instant now = Instant.now();

        // 1. Chỉ INSERT — không đọc (Read) trước để tránh Read-Modify-Write race condition
        String rawRefreshToken = user.getRole() == org.example.authservice.models.Role.CANDIDATE
                ? jwtUtil.generateCandidateRefreshToken(user.getId(), user.getEmail(), user.getRole())
                : jwtUtil.generateRefreshToken(user.getId(), user.getPhone(), user.getRole());

        RefreshToken refreshToken = RefreshToken.builder()
                .user(user)
                .token(rawRefreshToken)
                .expiresAt(now.plusMillis(refreshDuration))
                .build();

        RefreshToken savedToken = refreshTokenRepository.save(refreshToken);

        // 2. Dọn dẹp token thừa ngay lập tức bằng 1 query atomic
        try {
            int deleted = refreshTokenRepository.deleteOldTokensForUser(user.getId(), MAX_ACTIVE_TOKENS_PER_USER);
            if (deleted > 0) {
                System.out.println("[RefreshTokenService] Cleaned up " + deleted + " excess tokens for user " + user.getId());
            }
        } catch (Exception e) {
            // Không làm gián đoạn tiến trình login nếu cleanup thất bại tạm thời (ví dụ trùng lock)
            System.err.println("[RefreshTokenService] Failed to cleanup excess tokens: " + e.getMessage());
        }

        return savedToken;
    }

    /**
     * Kiểm tra token còn hạn không.
     * Nếu hết hạn thì xóa luôn và throw exception.
     */
    @Transactional
    public void verifyExpiration(RefreshToken token) {
        if (token.getExpiresAt().isBefore(Instant.now())) {
            refreshTokenRepository.delete(token);
            throw new RuntimeException(ErrorCode.REFRESH_TOKEN_EXPIRED.getMessage());
        }
    }

    /**
     * Tìm token theo string value.
     */
    public RefreshToken findByToken(String token) {
        return refreshTokenRepository.findByToken(token)
                .orElseThrow(() -> new RuntimeException(
                        ErrorCode.REFRESH_TOKEN_NOT_FOUND.getMessage()));
    }

    /**
     * Logout 1 thiết bị: xóa đúng token đó.
     */
    @Transactional
    public void findValidateAndDelete(String token, String userId) {
        RefreshToken refreshToken = refreshTokenRepository.findByToken(token)
                .orElseThrow(() -> new RuntimeException(
                        ErrorCode.REFRESH_TOKEN_NOT_FOUND.getMessage()));

        if (!refreshToken.getUser().getId().equals(userId)) {
            throw new RuntimeException(ErrorCode.FORBIDDEN.getMessage());
        }

        refreshTokenRepository.delete(refreshToken);
    }

    /**
     * Force logout tất cả thiết bị: xóa toàn bộ token của user.
     */
    @Transactional
    public void deleteAllByUser(Users user) {
        refreshTokenRepository.deleteByUser(user);
    }

    /**
     * Cleanup job: xóa token hết hạn mỗi ngày lúc 3 giờ sáng.
     * Tránh bảng refresh_tokens phình to theo thời gian.
     */
    @Scheduled(cron = "0 0 3 * * *", zone = "Asia/Ho_Chi_Minh")
    @Transactional
    public void cleanupExpiredTokens() {
        int deleted = refreshTokenRepository.deleteExpiredTokens(Instant.now());
        System.out.println("[RefreshTokenCleanup] Deleted " + deleted + " expired tokens");
    }
}