package org.example.recruitmentservice.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Getter;

import java.util.List;

/**
 * DTO trả về danh sách active positions cho chatbot-service.
 * chatbot-service dùng id list này để filter Qdrant JD collection.
 * openedAt dạng String (ISO) để tránh serialization issue qua HTTP.
 */
@Getter
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ActivePositionResponse {
    private Integer id;
    private String title;
    private String seniority;
    private List<String> skills;
    private Double minimumFitScore;
    private String openedAt;
}
