package org.example.apigateway.security;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;

@Component
public class JwtAuthenticationFilter implements GlobalFilter, Ordered {

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private RouterValidator routerValidator;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getURI().getPath();
        String uri = request.getURI().toString();

        // Nếu endpoint KHÔNG cần bảo mật → forward luôn
        if (!routerValidator.isSecured.test(request)) {
            System.out.println("Open endpoint, forwarding to downstream: " + path);
            return chain.filter(exchange)
                    .doOnSuccess(aVoid -> System.out.println("Successfully forwarded to: " + uri))
                    .doOnError(error -> System.out.println("Error forwarding to " + uri + ": " + error.getMessage()));
        }

        // Nếu CẦN bảo mật → xác thực JWT
        System.out.println("Secured endpoint, checking authentication: " + path);
        return handleAuthentication(exchange, chain);
    }

    private Mono<Void> handleAuthentication(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getURI().getPath();
        String uri = request.getURI().toString();

        // Check authorization header
        if (!request.getHeaders().containsKey(HttpHeaders.AUTHORIZATION)) {
            System.out.println("Missing authorization header for: " + path);
            return onError(exchange, ErrorCode.TOKEN_MISSING);
        }

        String authHeader = request.getHeaders().getFirst(HttpHeaders.AUTHORIZATION);
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            System.out.println("Invalid authorization header format for: " + path);
            return onError(exchange, ErrorCode.TOKEN_INVALID);
        }

        String token = authHeader.substring(7);
        try {
            JwtUtil.TokenValidationResult validationResult = jwtUtil.validateToken(token);

            if (!validationResult.isValid()) {
                if (validationResult.isExpired()) {
                    System.out.println("Token expired for: " + path);
                    return onError(exchange, ErrorCode.TOKEN_EXPIRED);
                }

                ErrorCode errorCode = mapToErrorCode(validationResult.getErrorCode());
                System.out.println("Token validation failed for: " + path);
                return onError(exchange, errorCode);
            }

            // Extract claims và thêm vào headers
            String role = jwtUtil.extractRole(token);
            String id = jwtUtil.extractId(token);
            boolean isCandidate = "CANDIDATE".equalsIgnoreCase(role);
            String phone = isCandidate ? null : jwtUtil.extractPhone(token);
            String email = isCandidate ? jwtUtil.extractEmail(token) : null;

            System.out.println("JWT valid - Id: " + id + ", Role: " + role
                    + (isCandidate ? ", Email: " + email : ", Phone: " + phone));

            // Kiểm tra phân quyền theo prefix route
            if (path.startsWith("/admin") || path.startsWith("/api/admin") || path.startsWith("/auth/admin")) {
                if (!"ADMIN".equalsIgnoreCase(role)) {
                    System.out.println("Access denied: Not an admin for path " + path);
                    return onError(exchange, ErrorCode.FORBIDDEN);
                }
            }

            if (path.startsWith("/hr")) {
                if (!"HR".equalsIgnoreCase(role)) {
                    System.out.println("Access denied: Not an HR user for path " + path);
                    return onError(exchange, ErrorCode.FORBIDDEN);
                }
            }

            System.out.println("Forwarding to downstream: " + uri);

            ServerHttpRequest.Builder requestBuilder = exchange.getRequest().mutate()
                    .header("X-User-Role", role)
                    .header("X-User-Id", id);
            if (phone != null) requestBuilder.header("X-User-Phone", phone);
            if (email != null) requestBuilder.header("X-User-Email", email);
            ServerHttpRequest modifiedRequest = requestBuilder.build();

            return chain.filter(exchange.mutate().request(modifiedRequest).build())
                    .doOnSuccess(aVoid -> System.out.println("Successfully forwarded to: " + uri))
                    .doOnError(error -> System.out.println("Error forwarding to " + uri + ": " + error.getMessage()));

        } catch (Exception e) {
            System.out.println("Token validation exception for " + path + ": " + e.getMessage());
            e.printStackTrace();
            return onError(exchange, ErrorCode.INTERNAL_SERVER_ERROR);
        }
    }

    /**
     * Map JWT validation error code to ErrorCode enum
     */
    private ErrorCode mapToErrorCode(String errorCode) {
        switch (errorCode) {
            case "INVALID_SIGNATURE":
            case "MALFORMED_TOKEN":
            case "INVALID_SUBJECT":
                return ErrorCode.TOKEN_INVALID;
            case "MISSING_CLAIM_ID":
            case "MISSING_CLAIM_PHONE":
            case "MISSING_CLAIM_EMAIL":
            case "MISSING_CLAIM_ROLE":
            case "MISSING_EXPIRATION":
                return ErrorCode.INVALID_REQUEST;
            default:
                return ErrorCode.TOKEN_INVALID;
        }
    }

    /**
     * Tạo error response theo chuẩn ApiResponse với ErrorCode
     */
    private Mono<Void> onError(ServerWebExchange exchange, ErrorCode errorCode) {
        ServerHttpResponse response = exchange.getResponse();

        response.setStatusCode(errorCode.getHttpStatus());
        response.getHeaders().setContentType(MediaType.APPLICATION_JSON);

        try {
            // Sử dụng ApiResponse từ common library
            ApiResponse<?> apiResponse = ApiResponse.builder()
                    .statusCode(errorCode.getCode())
                    .message(errorCode.getMessage())
                    .data(null)
                    .build();

            byte[] bytes = objectMapper.writeValueAsBytes(apiResponse);
            DataBuffer buffer = response.bufferFactory().wrap(bytes);

            return response.writeWith(Mono.just(buffer));

        } catch (Exception e) {
            // Fallback nếu serialization fails
            String fallbackJson = String.format(
                    "{\"status\":%d,\"message\":\"%s\",\"data\":null}",
                    errorCode.getCode(),
                    errorCode.getMessage());

            byte[] bytes = fallbackJson.getBytes(StandardCharsets.UTF_8);
            DataBuffer buffer = response.bufferFactory().wrap(bytes);

            return response.writeWith(Mono.just(buffer));
        }
    }

    @Override
    public int getOrder() {
        return -100;
    }
}