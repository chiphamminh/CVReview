package org.example.authservice.controller;

import org.example.authservice.dto.request.*;
import org.example.authservice.dto.response.*;
import org.example.authservice.models.RefreshToken;
import org.example.authservice.models.Role;
import org.example.authservice.models.Users;
import org.example.authservice.security.JwtUtil;
import org.example.authservice.services.AuthService;
import org.example.authservice.services.RefreshTokenService;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
public class AuthController {
    private final AuthService authService;
    private final RefreshTokenService refreshTokenService;
    private final JwtUtil jwtUtil;

    public AuthController(AuthService authService, RefreshTokenService refreshTokenService, JwtUtil jwtUtil) {
        this.authService = authService;
        this.refreshTokenService = refreshTokenService;
        this.jwtUtil = jwtUtil;
    }

    @PostMapping("/login")
    public ApiResponse<LoginData> login(@RequestBody LoginRequest loginRequest) {
        return authService.login(loginRequest);
    }

    @PostMapping("/logout")
    public ApiResponse<LogoutData> logout(
            @RequestHeader(value = "X-User-Id", required = true) String userId,
            @RequestHeader(value = "X-User-Phone", required = false) String userPhone,
            @RequestHeader(value = "X-User-Role", required = false) String userRole,
            @RequestBody LogoutRequest logoutRequest) {

        // LogoutRequest đã có cả accessToken và refreshToken rồi
        return authService.logout(userId, logoutRequest);
    }

    @PostMapping("/refresh-token")
    public ResponseEntity<ApiResponse<RefreshTokenResponse>> refreshToken(
            @RequestBody RefreshTokenRequest refreshTokenRequest) {
        try {
            RefreshToken refreshToken = refreshTokenService.findByToken(
                    refreshTokenRequest.getRefreshToken());
            refreshTokenService.verifyExpiration(refreshToken);

            Users user = refreshToken.getUser();
            String newAccessToken = user.getRole() == Role.CANDIDATE
                    ? jwtUtil.generateCandidateAccessToken(user.getId(), user.getEmail(), user.getRole())
                    : jwtUtil.generateAccessToken(user.getId(), user.getPhone(), user.getRole());

            RefreshTokenResponse responseData = new RefreshTokenResponse(
                    refreshToken.getToken(), newAccessToken);

            return ResponseEntity.ok(new ApiResponse<>(
                    ErrorCode.SUCCESS.getCode(), ErrorCode.SUCCESS.getMessage(), responseData));

        } catch (RuntimeException e) {
            ErrorCode errorCode;
            if (e.getMessage().contains("expired")) {
                errorCode = ErrorCode.REFRESH_TOKEN_EXPIRED;
            } else if (e.getMessage().contains("not found")) {
                errorCode = ErrorCode.REFRESH_TOKEN_NOT_FOUND;
            } else {
                errorCode = ErrorCode.REFRESH_TOKEN_INVALID;
            }
            return ResponseEntity.status(errorCode.getHttpStatus())
                    .body(new ApiResponse<>(errorCode.getCode(), errorCode.getMessage(), null));
        }
    }

    // ─── Candidate auth endpoints ──────────────────────────────────────────────

    @PostMapping("/candidate/register")
    public ApiResponse<Void> candidateRegister(@RequestBody CandidateRegisterRequest request) {
        return authService.candidateRegister(request);
    }

    @PostMapping("/candidate/verify-register")
    public ApiResponse<Void> verifyRegister(@RequestBody VerifyRegisterRequest request) {
        return authService.verifyRegistration(request);
    }

    @PostMapping("/candidate/login")
    public ApiResponse<LoginData> candidateLogin(@RequestBody CandidateLoginRequest request) {
        return authService.candidateLogin(request);
    }

    @PostMapping("/candidate/forgot-password")
    public ApiResponse<Void> forgotPassword(@RequestBody ForgotPasswordRequest request) {
        return authService.forgotPassword(request);
    }

    @PostMapping("/candidate/verify-reset-otp")
    public ApiResponse<VerifyResetOtpResponse> verifyResetOtp(@RequestBody VerifyOtpRequest request) {
        return authService.verifyResetOtp(request);
    }

    @PostMapping("/candidate/reset-password")
    public ApiResponse<Void> resetPassword(@RequestBody ResetPasswordRequest request) {
        return authService.resetPassword(request);
    }

    @GetMapping("/user-detail")
    public ApiResponse<Userdata> getUserDetail(
            @RequestHeader(value = "X-User-Id", required = true) String userId,
            @RequestHeader(value = "Authorization", required = true) String authorizationHeader,
            @RequestHeader(value = "X-User-Phone", required = false) String userPhone,
            @RequestHeader(value = "X-User-Role", required = false) String userRole) {
        String token = authorizationHeader.replace("Bearer ", "");

        return authService.getUserDetail(userId, token);
    }
}
