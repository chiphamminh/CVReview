package org.example.authservice.services;

import org.example.authservice.dto.request.*;
import org.example.authservice.dto.response.LoginData;
import org.example.authservice.dto.response.LogoutData;
import org.example.authservice.dto.response.Userdata;
import org.example.authservice.dto.response.VerifyResetOtpResponse;
import org.example.authservice.models.OtpPurpose;
import org.example.authservice.models.RefreshToken;
import org.example.authservice.models.Role;
import org.example.authservice.models.Users;
import org.example.authservice.repository.UserRepository;
import org.example.authservice.security.JwtUtil;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.exception.CustomException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
public class AuthService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private RefreshTokenService refreshTokenService;

    @Autowired
    private OtpService otpService;

    public ApiResponse<LoginData> login(LoginRequest loginRequest) {
        try {
            // Validate input
            if (loginRequest.getPhone() == null || loginRequest.getPhone().trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(),
                        "Phone is required", null);
            }
            if (loginRequest.getPassword() == null || loginRequest.getPassword().trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(),
                        "Password is required", null);
            }

            Users user = userRepository.findByPhone(loginRequest.getPhone());
            if (user == null) {
                return new ApiResponse<>(ErrorCode.USER_NOT_FOUND.getCode(),
                        ErrorCode.USER_NOT_FOUND.getMessage(), null);
            }

            if (!isPasswordValid(user, loginRequest.getPassword())) {
                return new ApiResponse<>(ErrorCode.INVALID_CREDENTIALS.getCode(),
                        ErrorCode.INVALID_CREDENTIALS.getMessage(), null);
            }

            // Upgrade password hash async — không block login
            if (needsPasswordUpgrade(user)) {
                upgradePasswordAsync(user, loginRequest.getPassword());
            }

            // Tạo token — chỉ INSERT, không DELETE → không deadlock
            String accessToken = jwtUtil.generateAccessToken(
                    user.getId(), user.getPhone(), user.getRole());
            RefreshToken refreshToken = refreshTokenService.createRefreshToken(user);

            if (accessToken == null) {
                return new ApiResponse<>(ErrorCode.JWT_GENERATION_FAILED.getCode(),
                        ErrorCode.JWT_GENERATION_FAILED.getMessage(), null);
            }

            LoginData.AccountInfo accountInfo = new LoginData.AccountInfo(
                    user.getId(),
                    user.getName(),
                    user.getEmail(),
                    user.getPhone(),
                    user.getRole(),
                    user.getCreatedAt());

            LoginData loginData = new LoginData(
                    accessToken,
                    refreshToken.getToken(),
                    accountInfo);

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "Welcome to CV Review System", loginData);

        } catch (Exception e) {
            System.err.println("Login failed: " + e.getMessage());
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.UNAUTHORIZED.getCode(),
                    "Login failed: " + e.getMessage(), null);
        }
    }

    public ApiResponse<LogoutData> logout(String userId, LogoutRequest logoutRequest) {
        try {
            if (userId == null || userId.trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(),
                        "User ID is required", null);
            }

            // Xóa đúng refresh token của thiết bị này
            if (logoutRequest.getRefreshToken() != null
                    && !logoutRequest.getRefreshToken().trim().isEmpty()) {
                try {
                    refreshTokenService.findValidateAndDelete(
                            logoutRequest.getRefreshToken(), userId);
                } catch (RuntimeException e) {
                    if (e.getMessage().equals(ErrorCode.FORBIDDEN.getMessage())) {
                        return new ApiResponse<>(ErrorCode.FORBIDDEN.getCode(),
                                "Refresh token does not belong to this user", null);
                    }
                    // Token không tồn tại hoặc đã bị xóa → vẫn coi là logout thành công
                    System.out.println("Refresh token not found or already deleted: "
                            + e.getMessage());
                }
            }

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "Logout successful", new LogoutData("Goodbye"));

        } catch (Exception e) {
            System.err.println("Logout error for user " + userId + ": " + e.getMessage());
            return new ApiResponse<>(ErrorCode.UNAUTHORIZED.getCode(),
                    "Logout failed: " + e.getMessage(), null);
        }
    }

    @Transactional
    public ApiResponse<Userdata> getUserDetail(String userId, String token) {
        try {
            if (userId == null || userId.trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(),
                        "User ID is required", null);
            }

            Optional<Users> user = userRepository.findById(userId);
            if (user.isEmpty()) {
                return new ApiResponse<>(ErrorCode.USER_NOT_FOUND.getCode(),
                        ErrorCode.USER_NOT_FOUND.getMessage(), null);
            }
            Users foundUser = user.get();

            Userdata.UserInfo userInfo = new Userdata.UserInfo(
                    foundUser.getId(),
                    foundUser.getName(),
                    foundUser.getEmail(),
                    foundUser.getPhone(),
                    foundUser.getRole(),
                    foundUser.getCreatedAt());

            Userdata userdata = new Userdata();
            userdata.setAccessToken(token);
            userdata.setAccount(userInfo);

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "User detail fetched successfully", userdata);

        } catch (Exception e) {
            System.err.println("Failed to get user detail: " + e.getMessage());
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.USER_NOT_FOUND.getCode(),
                    "User not found", null);
        }
    }

    // ─── Candidate auth ───────────────────────────────────────────────────────

    @Transactional
    public ApiResponse<Void> candidateRegister(CandidateRegisterRequest request) {
        if (request.getEmail() == null || request.getEmail().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Email is required", null);
        if (request.getName() == null || request.getName().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Name is required", null);
        if (request.getPassword() == null || request.getPassword().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Password is required", null);

        String email = request.getEmail().toLowerCase().trim();

        if (userRepository.findByEmail(email) != null)
            return new ApiResponse<>(ErrorCode.DUPLICATE_EMAIL.getCode(),
                    ErrorCode.DUPLICATE_EMAIL.getMessage(), null);

        otpService.generateAndSend(email, OtpPurpose.REGISTRATION);
        return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                "OTP sent to " + email + ". Please verify to complete registration.", null);
    }

    @Transactional
    public ApiResponse<Void> verifyRegistration(VerifyRegisterRequest request) {
        if (request.getEmail() == null || request.getEmail().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Email is required", null);
        if (request.getOtp() == null || request.getOtp().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "OTP is required", null);
        if (request.getName() == null || request.getName().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Name is required", null);
        if (request.getPassword() == null || request.getPassword().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Password is required", null);

        String email = request.getEmail().toLowerCase().trim();

        // Re-check duplicate in case of race condition between OTP send and verify
        if (userRepository.findByEmail(email) != null)
            return new ApiResponse<>(ErrorCode.DUPLICATE_EMAIL.getCode(),
                    ErrorCode.DUPLICATE_EMAIL.getMessage(), null);

        try {
            otpService.verifyOtp(email, request.getOtp(), OtpPurpose.REGISTRATION);
        } catch (CustomException e) {
            return new ApiResponse<>(e.getErrorCode().getCode(), e.getMessage(), null);
        }

        Users candidate = Users.builder()
                .name(request.getName().trim())
                .email(email)
                .password(passwordEncoder.encode(request.getPassword()))
                .role(Role.CANDIDATE)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        userRepository.save(candidate);

        return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                "Registration successful. Please log in.", null);
    }

    public ApiResponse<LoginData> candidateLogin(CandidateLoginRequest request) {
        if (request.getEmail() == null || request.getEmail().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Email is required", null);
        if (request.getPassword() == null || request.getPassword().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Password is required", null);

        try {
            String email = request.getEmail().toLowerCase().trim();
            Users user = userRepository.findByEmail(email);
            if (user == null)
                return new ApiResponse<>(ErrorCode.USER_NOT_FOUND.getCode(),
                        ErrorCode.USER_NOT_FOUND.getMessage(), null);

            if (user.getRole() != Role.CANDIDATE)
                return new ApiResponse<>(ErrorCode.FORBIDDEN.getCode(),
                        "This login is for candidates only", null);

            if (!isPasswordValid(user, request.getPassword()))
                return new ApiResponse<>(ErrorCode.INVALID_CREDENTIALS.getCode(),
                        ErrorCode.INVALID_CREDENTIALS.getMessage(), null);

            if (needsPasswordUpgrade(user))
                upgradePasswordAsync(user, request.getPassword());

            String accessToken = jwtUtil.generateCandidateAccessToken(user.getId(), user.getEmail(), user.getRole());
            RefreshToken refreshToken = refreshTokenService.createRefreshToken(user);

            LoginData.AccountInfo accountInfo = new LoginData.AccountInfo(
                    user.getId(), user.getName(), user.getEmail(),
                    user.getPhone(), user.getRole(), user.getCreatedAt());

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "Welcome to CV Review System", new LoginData(accessToken, refreshToken.getToken(), accountInfo));

        } catch (Exception e) {
            return new ApiResponse<>(ErrorCode.UNAUTHORIZED.getCode(),
                    "Login failed: " + e.getMessage(), null);
        }
    }

    public ApiResponse<Void> forgotPassword(ForgotPasswordRequest request) {
        if (request.getEmail() == null || request.getEmail().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Email is required", null);

        String email = request.getEmail().toLowerCase().trim();
        Users user = userRepository.findByEmail(email);

        // Always return success to prevent email enumeration
        if (user == null || user.getRole() != Role.CANDIDATE)
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "If this email is registered, an OTP will be sent.", null);

        otpService.generateAndSend(email, OtpPurpose.RESET_PASSWORD);
        return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                "If this email is registered, an OTP will be sent.", null);
    }

    public ApiResponse<VerifyResetOtpResponse> verifyResetOtp(VerifyOtpRequest request) {
        if (request.getEmail() == null || request.getEmail().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Email is required", null);
        if (request.getOtp() == null || request.getOtp().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "OTP is required", null);

        try {
            String resetToken = otpService.verifyOtp(
                    request.getEmail().toLowerCase().trim(),
                    request.getOtp(),
                    OtpPurpose.RESET_PASSWORD);
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "OTP verified", new VerifyResetOtpResponse(resetToken));
        } catch (CustomException e) {
            return new ApiResponse<>(e.getErrorCode().getCode(), e.getMessage(), null);
        }
    }

    @Transactional
    public ApiResponse<Void> resetPassword(ResetPasswordRequest request) {
        if (request.getResetToken() == null || request.getResetToken().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Reset token is required", null);
        if (request.getNewPassword() == null || request.getNewPassword().isBlank())
            return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "New password is required", null);

        try {
            String email = otpService.consumeResetToken(request.getResetToken());
            Users user = userRepository.findByEmail(email);
            if (user == null)
                return new ApiResponse<>(ErrorCode.USER_NOT_FOUND.getCode(),
                        ErrorCode.USER_NOT_FOUND.getMessage(), null);

            user.setPassword(passwordEncoder.encode(request.getNewPassword()));
            user.setUpdatedAt(LocalDateTime.now());
            userRepository.save(user);

            // Invalidate all sessions after password reset
            refreshTokenService.deleteAllByUser(user);

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(),
                    "Password reset successful. Please log in with your new password.", null);
        } catch (CustomException e) {
            return new ApiResponse<>(e.getErrorCode().getCode(), e.getMessage(), null);
        }
    }

    // ─── Helper methods ───────────────────────────────────────────────────────

    private boolean isPasswordValid(Users user, String rawPassword) {
        String pwd = user.getPassword();
        if (pwd.startsWith("$2a$") || pwd.startsWith("$2b$") || pwd.startsWith("$2y$")) {
            return passwordEncoder.matches(rawPassword, pwd);
        }
        return pwd.equals(rawPassword);
    }

    private boolean needsPasswordUpgrade(Users user) {
        String pwd = user.getPassword();
        return !pwd.startsWith("$2a$") && !pwd.startsWith("$2b$") && !pwd.startsWith("$2y$");
    }

    @Async
    @Transactional
    public void upgradePasswordAsync(Users user, String rawPassword) {
        user.setPassword(passwordEncoder.encode(rawPassword));
        userRepository.save(user);
    }

    @Transactional
    public ApiResponse<Userdata> registerHr(RegisterHrRequest request) {
        try {
            if (request.getPhone() == null || request.getPhone().trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Phone is required", null);
            }
            if (request.getPassword() == null || request.getPassword().trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Password is required", null);
            }
            if (request.getName() == null || request.getName().trim().isEmpty()) {
                return new ApiResponse<>(ErrorCode.MISSING_REQUIRED_FIELD.getCode(), "Name is required", null);
            }

            // Check duplicate phone
            if (userRepository.findByPhone(request.getPhone()) != null) {
                return new ApiResponse<>(ErrorCode.DUPLICATE_PHONE.getCode(), ErrorCode.DUPLICATE_PHONE.getMessage(),
                        null);
            }

            Users hrUser = Users.builder()
                    .name(request.getName())
                    .email(request.getEmail() != null ? request.getEmail() : "")
                    .phone(request.getPhone())
                    .password(passwordEncoder.encode(request.getPassword()))
                    .role(Role.HR)
                    .createdAt(LocalDateTime.now())
                    .updatedAt(LocalDateTime.now())
                    .build();

            Users savedUser = userRepository.save(hrUser);

            Userdata.UserInfo userInfo = new Userdata.UserInfo(
                    savedUser.getId(),
                    savedUser.getName(),
                    savedUser.getEmail(),
                    savedUser.getPhone(),
                    savedUser.getRole(),
                    savedUser.getCreatedAt());

            Userdata userdata = new Userdata();
            userdata.setAccount(userInfo);

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "HR account created successfully", userdata);

        } catch (Exception e) {
            System.err.println("Register HR failed: " + e.getMessage());
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Register HR failed: " + e.getMessage(),
                    null);
        }
    }
}