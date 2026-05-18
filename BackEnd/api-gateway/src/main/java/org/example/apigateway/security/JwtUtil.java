package org.example.apigateway.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.security.SignatureException;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;

@Component
public class JwtUtil {

    private final SecretKey key;
    private final String expectedSubject = "CV Review";

    public JwtUtil(@Value("${jwt.secret}") String secret) {
        if (secret.length() < 32) {
            throw new IllegalArgumentException("JWT secret must be at least 32 characters long");
        }
        this.key = Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Validate token and return detailed result with specific error codes
     */
    public TokenValidationResult validateToken(String token) {
        try {
            Claims claims = extractAllClaims(token);

            // Check subject
            if (!expectedSubject.equals(claims.getSubject())) {
                System.out.println("Invalid subject in token");
                return TokenValidationResult.invalid("INVALID_SUBJECT", "Token subject does not match expected value");
            }

            // Check expiration
            Date expiration = claims.getExpiration();
            if (expiration == null) {
                System.out.println("Missing expiration in token");
                return TokenValidationResult.invalid("MISSING_EXPIRATION", "Token does not contain expiration date");
            }

            if (expiration.before(new Date())) {
                System.out.println("Token expired at: " + expiration);
                return TokenValidationResult.expired("Token has expired. Please refresh your token or login again");
            }

            // Check required claims
            String id = claims.get("Id", String.class);
            String role = claims.get("Role", String.class);
            String phone = claims.get("Phone", String.class);
            String email = claims.get("Email", String.class);

            if (id == null || id.isEmpty()) {
                System.out.println("Missing Id claim");
                return TokenValidationResult.invalid("MISSING_CLAIM_ID", "Token is missing required claim: Id");
            }
            if (role == null || role.isEmpty()) {
                System.out.println("Missing Role claim");
                return TokenValidationResult.invalid("MISSING_CLAIM_ROLE", "Token is missing required claim: Role");
            }
            // CANDIDATE tokens use Email claim; HR/ADMIN tokens use Phone claim
            boolean isCandidate = "CANDIDATE".equalsIgnoreCase(role);
            if (!isCandidate && (phone == null || phone.isEmpty())) {
                System.out.println("Missing Phone claim for non-candidate role");
                return TokenValidationResult.invalid("MISSING_CLAIM_PHONE", "Token is missing required claim: Phone");
            }
            if (isCandidate && (email == null || email.isEmpty())) {
                System.out.println("Missing Email claim for CANDIDATE role");
                return TokenValidationResult.invalid("MISSING_CLAIM_EMAIL", "Token is missing required claim: Email");
            }

            System.out.println("Token validated successfully for user: " + (isCandidate ? email : phone));
            return TokenValidationResult.valid();

        } catch (ExpiredJwtException e) {
            System.out.println("Token expired: " + e.getMessage());
            return TokenValidationResult.expired("Token has expired. Please refresh your token or login again");
        } catch (SignatureException e) {
            System.out.println("Invalid token signature: " + e.getMessage());
            return TokenValidationResult.invalid("INVALID_SIGNATURE", "Token signature verification failed");
        } catch (MalformedJwtException e) {
            System.out.println("Malformed token: " + e.getMessage());
            return TokenValidationResult.invalid("MALFORMED_TOKEN", "Token format is invalid");
        } catch (Exception e) {
            System.out.println("Token validation error: " + e.getMessage());
            return TokenValidationResult.invalid("INVALID_TOKEN", "Token validation failed: " + e.getMessage());
        }
    }

    /**
     * Extract all claims from token
     * This will throw exceptions if token is invalid
     */
    private Claims extractAllClaims(String token) throws JwtException {
        return Jwts.parser()
                .verifyWith(key)
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    /**
     * Extract user ID from valid token
     */
    public String extractId(String token) {
        try {
            Claims claims = extractAllClaims(token);
            return claims.get("Id", String.class);
        } catch (Exception e) {
            throw new JwtException("Failed to extract Id from token: " + e.getMessage());
        }
    }

    /**
     * Extract phone from valid token
     */
    public String extractPhone(String token) {
        try {
            Claims claims = extractAllClaims(token);
            return claims.get("Phone", String.class);
        } catch (Exception e) {
            throw new JwtException("Failed to extract Phone from token: " + e.getMessage());
        }
    }

    /**
     * Extract email from valid token (CANDIDATE role)
     */
    public String extractEmail(String token) {
        try {
            Claims claims = extractAllClaims(token);
            return claims.get("Email", String.class);
        } catch (Exception e) {
            throw new JwtException("Failed to extract Email from token: " + e.getMessage());
        }
    }

    /**
     * Extract role from valid token
     */
    public String extractRole(String token) {
        try {
            Claims claims = extractAllClaims(token);
            return claims.get("Role", String.class);
        } catch (Exception e) {
            throw new JwtException("Failed to extract Role from token: " + e.getMessage());
        }
    }

    /**
     * Result class for token validation with detailed error information
     */
    @Getter
    public static class TokenValidationResult {
        private final boolean valid;
        private final boolean expired;
        private final String errorCode;
        private final String errorMessage;

        private TokenValidationResult(boolean valid, boolean expired, String errorCode, String errorMessage) {
            this.valid = valid;
            this.expired = expired;
            this.errorCode = errorCode;
            this.errorMessage = errorMessage;
        }

        public static TokenValidationResult valid() {
            return new TokenValidationResult(true, false, null, null);
        }

        public static TokenValidationResult expired(String message) {
            return new TokenValidationResult(false, true, "TOKEN_EXPIRED", message);
        }

        public static TokenValidationResult invalid(String errorCode, String message) {
            return new TokenValidationResult(false, false, errorCode, message);
        }

    }
}