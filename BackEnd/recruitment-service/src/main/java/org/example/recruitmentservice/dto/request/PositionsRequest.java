package org.example.recruitmentservice.dto.request;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@Builder
public class PositionsRequest {
    private String title;
    private String seniority;
    private List<String> skills;
    private MultipartFile file;
}

