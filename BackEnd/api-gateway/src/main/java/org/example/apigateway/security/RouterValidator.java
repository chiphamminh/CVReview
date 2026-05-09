package org.example.apigateway.security;

import lombok.Getter;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.Predicate;

@Component
public class RouterValidator {

    /**
     * Danh sách các endpoints KHÔNG cần JWT authentication
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
            "/chatbot/health/live"
    );

    /**
     * Predicate để kiểm tra xem một request có cần authentication hay không
     *
     * @return true nếu endpoint cần JWT authentication, false nếu là open endpoint
     */
    public Predicate<ServerHttpRequest> isSecured = request -> {
        String path = request.getURI().getPath();

        // Kiểm tra xem path có chính xác bằng bất kỳ endpoint nào trong danh sách public hay không
        boolean isOpenEndpoint = openEndpoints.contains(path);

        if (isOpenEndpoint) {
            return false;
        }

        // Tất cả các endpoint khác đều cần authentication
        return true;
    };
}