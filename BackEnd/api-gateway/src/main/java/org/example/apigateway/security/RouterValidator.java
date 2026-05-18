package org.example.apigateway.security;

import lombok.Getter;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.Predicate;

@Component
public class RouterValidator {

    /**
     * Các endpoints khớp chính xác — không cần JWT.
     */
    @Getter
    private static final List<String> openEndpoints = List.of(
            "/auth/login",
            "/auth/register",
            "/auth/refresh-token",
            "/auth/candidate/register",
            "/auth/candidate/verify-register",
            "/auth/candidate/login",
            "/auth/candidate/forgot-password",
            "/auth/candidate/verify-reset-otp",
            "/auth/candidate/reset-password",
            "/actuator/health",
            "/actuator/info",
            "/chatbot/health",
            "/chatbot/health/ready",
            "/chatbot/health/live"
    );

    /**
     * Các path prefix — bất kỳ path nào bắt đầu bằng các giá trị này đều không cần JWT.
     * Dùng cho endpoints có dynamic segment (e.g. /positions/jd/{id}/text).
     */
    private static final List<String> openPrefixes = List.of(
            "/positions/jd/"
    );

    /**
     * Predicate kiểm tra request có cần authentication không.
     * Trả về true nếu cần JWT, false nếu là open endpoint.
     */
    public Predicate<ServerHttpRequest> isSecured = request -> {
        String path = request.getURI().getPath();
        String method = request.getMethod().name();

        if (path.equals("/positions") && method.equalsIgnoreCase("GET")) {
            return false;
        }

        boolean isExactMatch = openEndpoints.contains(path);
        boolean isPrefixMatch = openPrefixes.stream().anyMatch(path::startsWith);
        return !isExactMatch && !isPrefixMatch;
    };
}