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
            "/actuator/health",
            "/actuator/info",
            "/chatbot/health",
            "/chatbot/health/ready",
            "/chatbot/health/live",
            "/positions"
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
        boolean isExactMatch = openEndpoints.contains(path);
        boolean isPrefixMatch = openPrefixes.stream().anyMatch(path::startsWith);
        return !isExactMatch && !isPrefixMatch;
    };
}