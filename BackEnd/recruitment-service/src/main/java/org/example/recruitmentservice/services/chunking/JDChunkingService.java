package org.example.recruitmentservice.services.chunking;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.recruitmentservice.dto.request.JDChunkPayload;
import org.example.recruitmentservice.services.chunking.config.ChunkingConfig;
import org.example.recruitmentservice.utils.TextUtils;
import org.example.recruitmentservice.utils.PositionUtils;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Dedicated chunking service for Job Description Markdown text.
 * Kept separate from CV ChunkingService (SRP) — JDs require no metadata
 * extraction
 * (Gemini, skills, experience years, etc.) and follow a different section
 * schema.
 *
 * Strategy:
 * 1. Split text on Markdown H1/H2 headers (# / ##) into named sections.
 * 2. If a section still exceeds maxTokens, split further by paragraphs.
 * 3. Each chunk carries positionId for Small-to-Big parent lookup in Qdrant.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class JDChunkingService {

    private final ChunkingConfig config;
    private final TextUtils textUtils;

    private static final Pattern HEADER_PATTERN = Pattern.compile(
            "(?:^|\\n)\\s*(#{1,2})\\s+([^#\\n\\r]+?)(?=\\s*\\n|$)",
            Pattern.MULTILINE);

    /**
     * Chunks a JD Markdown text into section-based {@link JDChunkPayload} list.
     *
     * @param positionId    the owning position's DB id
     * @param positionTitle position title (e.g. \"Senior Fullstack Engineer\")
     * @param seniority     position seniority level
     * @param jdMarkdown    the Markdown text returned by LlamaParse
     * @return list of chunks; never null, may be empty if input is blank
     */
    public List<JDChunkPayload> chunk(Integer positionId, String positionTitle,
            String seniority, String jdMarkdown) {
        if (jdMarkdown == null || jdMarkdown.isBlank()) {
            log.warn("[JDChunking] Empty JD text for position {}", positionId);
            return Collections.emptyList();
        }

        String formattedTitle = PositionUtils.formatPositionTitle(seniority, positionTitle);

        try {
            String normalized = normalize(jdMarkdown);
            List<RawSection> sections = extractSections(normalized);

            if (sections.isEmpty()) {
                log.info("[JDChunking] No headers found for position {}, treating as single chunk", positionId);
                return List.of(buildPayload(positionId, formattedTitle, seniority, "FULL_TEXT", 0, normalized));
            }

            List<JDChunkPayload> result = new ArrayList<>();
            int globalIndex = 0;

            for (RawSection section : sections) {
                int tokens = textUtils.estimateTokensFromWords(textUtils.countWords(section.text));

                if (tokens <= config.getMaxTokens()) {
                    result.add(buildPayload(
                            positionId, formattedTitle, seniority,
                            section.name, globalIndex++, section.text));
                } else {
                    log.debug("[JDChunking] Section '{}' exceeds maxTokens ({} tokens), splitting by paragraph",
                            section.name, tokens);
                    List<JDChunkPayload> sub = splitByParagraph(
                            positionId, formattedTitle, seniority, section.name, section.text, globalIndex);
                    result.addAll(sub);
                    globalIndex += sub.size();
                }
            }

            log.info("[JDChunking] Position {} → {} chunks generated", positionId, result.size());
            return result;

        } catch (Exception e) {
            log.error("[JDChunking] Failed to chunk JD for position {}: {}", positionId, e.getMessage(), e);
            return Collections.emptyList();
        }
    }

    // -------------------------------------------------------
    // Private Helpers
    // -------------------------------------------------------

    /** Extracts named sections by splitting on H1/H2 Markdown headers. */
    private List<RawSection> extractSections(String text) {
        Matcher matcher = HEADER_PATTERN.matcher(text);
        List<int[]> boundaries = new ArrayList<>(); // [headerStart, contentStart]
        List<String> names = new ArrayList<>();

        while (matcher.find()) {
            String headerName = matcher.group(2).trim();
            if (headerName.length() > 120)
                continue; // skip suspiciously long headers
            boundaries.add(new int[] { matcher.start(), matcher.end() });
            names.add(normalizeHeaderName(headerName));
        }

        if (boundaries.isEmpty()) {
            return Collections.emptyList();
        }

        List<RawSection> sections = new ArrayList<>();
        for (int i = 0; i < boundaries.size(); i++) {
            int contentStart = boundaries.get(i)[1];
            int contentEnd = (i + 1 < boundaries.size())
                    ? boundaries.get(i + 1)[0]
                    : text.length();

            String sectionText = text.substring(contentStart, contentEnd).trim();
            if (!sectionText.isBlank()) {
                sections.add(new RawSection(names.get(i), sectionText));
            }
        }
        return sections;
    }

    /**
     * Splits an oversized section by double-newline paragraphs.
     * Falls back to keeping the entire section as one chunk if it cannot be split
     * further.
     */
    private List<JDChunkPayload> splitByParagraph(Integer positionId, String positionTitle,
            String seniority,
            String sectionName, String sectionText, int startIndex) {
        List<JDChunkPayload> chunks = new ArrayList<>();
        String[] paragraphs = sectionText.split("\\n\\n+");

        if (paragraphs.length <= 1) {
            log.warn("[JDChunking] Section '{}' cannot be split further, keeping as oversized chunk", sectionName);
            chunks.add(buildPayload(positionId, positionTitle, seniority, sectionName, startIndex, sectionText));
            return chunks;
        }

        StringBuilder buffer = new StringBuilder();
        int bufferWords = 0;
        int chunkIdx = startIndex;

        for (String paragraph : paragraphs) {
            paragraph = paragraph.trim();
            if (paragraph.isBlank())
                continue;

            int paraWords = textUtils.countWords(paragraph);
            int bufferTokens = textUtils.estimateTokensFromWords(bufferWords);
            int paraTokens = textUtils.estimateTokensFromWords(paraWords);

            if (bufferTokens + paraTokens <= config.getMaxTokens()) {
                if (!buffer.isEmpty())
                    buffer.append("\n\n");
                buffer.append(paragraph);
                bufferWords += paraWords;
            } else {
                if (!buffer.isEmpty()) {
                    chunks.add(buildPayload(
                            positionId, positionTitle, seniority,
                            sectionName, chunkIdx++, buffer.toString()));
                }
                buffer = new StringBuilder(paragraph);
                bufferWords = paraWords;
            }
        }

        if (!buffer.isEmpty()) {
            chunks.add(buildPayload(
                    positionId, positionTitle, seniority,
                    sectionName, chunkIdx, buffer.toString()));
        }

        return chunks;
    }

    private JDChunkPayload buildPayload(Integer positionId, String positionTitle,
            String seniority,
            String sectionName, int chunkIndex, String text) {
        int words = textUtils.countWords(text);
        int tokens = textUtils.estimateTokensFromWords(words);
        return JDChunkPayload.builder()
                .positionId(positionId)
                .positionTitle(positionTitle)
                .seniority(seniority)
                .sectionName(sectionName)
                .chunkIndex(chunkIndex)
                .chunkText(text)
                .words(words)
                .tokensEstimate(tokens)
                .build();
    }

    /**
     * Normalises a raw Markdown header string to a consistent uppercase key (e.g.
     * "Job Requirements" → "JOB_REQUIREMENTS").
     */
    private String normalizeHeaderName(String raw) {
        return raw.toUpperCase()
                .replaceAll("[^A-Z0-9]+", "_")
                .replaceAll("^_|_$", "");
    }

    /**
     * Light normalization for JD text — strips carriage returns and collapses
     * excessive blank lines.
     */
    private String normalize(String text) {
        return text
                .replaceAll("\\r\\n", "\n")
                .replaceAll("\\r", "\n")
                .replaceAll("\\n{3,}", "\n\n")
                .trim();
    }

    /** Internal value-holder for extracted sections before building payloads. */
    private record RawSection(String name, String text) {
    }
}
