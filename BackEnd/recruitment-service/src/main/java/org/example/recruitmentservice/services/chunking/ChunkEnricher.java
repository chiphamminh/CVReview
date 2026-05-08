package org.example.recruitmentservice.services.chunking;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.services.metadata.model.CVMetadata;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class ChunkEnricher {

    /**
     * Enrich chunk text với candidate context header
     */
    public String enrichChunkText(
            CandidateCV cv,
            String section,
            String rawText,
            CVMetadata metadata) {
        StringBuilder enriched = new StringBuilder();

        // Build context header
        enriched.append("CANDIDATE: ").append(cv.getName() != null ? cv.getName() : "Unknown").append("\n");

        if (cv.getPosition() != null && cv.getPosition().getTitle() != null) {
            enriched.append("POSITION: ").append(cv.getPosition().getTitle()).append("\n");
        }

        if (metadata.getExperienceYears() != null && metadata.getExperienceYears() > 0) {
            enriched.append("EXPERIENCE: ")
                    .append(metadata.getExperienceYears())
                    .append(" years (")
                    .append(metadata.getSeniorityLevel())
                    .append(")\n");
        }

        // Section delimiter
        enriched.append("\n===== ").append(section.toUpperCase()).append(" =====\n");

        // Original content
        enriched.append(rawText);

        return enriched.toString();
    }

    /**
     * Tạo executive summary từ full CV (thay thế FullText chunk)
     */
    public String createExecutiveSummary(
            CandidateCV cv,
            String fullText,
            CVMetadata metadata) {
        StringBuilder summary = new StringBuilder();

        summary.append("CANDIDATE: ").append(cv.getName() != null ? cv.getName() : "Unknown").append("\n");

        if (cv.getPosition() != null && cv.getPosition().getTitle() != null) {
            summary.append("POSITION: ").append(cv.getPosition().getTitle()).append("\n");
        }

        if (metadata.getExperienceYears() != null) {
            summary.append("EXPERIENCE: ")
                    .append(metadata.getExperienceYears())
                    .append(" years (")
                    .append(metadata.getSeniorityLevel())
                    .append(")\n");
        }

        // Top skills (limit to 10)
        if (metadata.getSkills() != null && !metadata.getSkills().isEmpty()) {
            summary.append("TOP SKILLS: ");
            summary.append(String.join(", ",
                    metadata.getSkills().stream().limit(10).toList()));
            summary.append("\n");
        }

        // Companies worked at
        if (metadata.getCompanies() != null && !metadata.getCompanies().isEmpty()) {
            summary.append("COMPANIES: ");
            summary.append(String.join(", ", metadata.getCompanies()));
            summary.append("\n");
        }

        // Education
        if (metadata.getDegrees() != null && !metadata.getDegrees().isEmpty()) {
            summary.append("EDUCATION: ");
            summary.append(String.join(", ", metadata.getDegrees()));
            summary.append("\n");
        }

        return summary.toString();
    }
}