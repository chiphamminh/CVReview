package org.example.recruitmentservice.client;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.exception.CustomException;
import org.example.recruitmentservice.config.RabbitMQConfig;
import org.example.recruitmentservice.dto.request.CVUploadEvent;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.services.ProcessingBatchService;
import org.example.recruitmentservice.services.StorageService;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.io.File;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Component
@RequiredArgsConstructor
public class LlamaParseClient {

    @Value("${llama-parse.api-key}")
    private String apiKey;

    private final RestTemplate restTemplate = new RestTemplate();
    private final CandidateCVRepository candidateCVRepository;
    private final StorageService storageService;
    private final ProcessingBatchService processingBatchService;
    private final RabbitTemplate rabbitTemplate;

    /**
     * Parse JD từ file path (temp file đã download từ Drive)
     */
    public String parseJD(String filePath) {
        try {
            log.debug("API Key: {}", (apiKey != null ? "exists" : "null"));
            log.debug("File path: {}", filePath);

            String jobId = uploadFileForJD(filePath);
            log.debug("Job ID: {}", jobId);

            // Poll result
            String parsedText = pollResult(jobId);
            log.debug("Parse completed!");

            return parsedText;

        } catch (Exception e) {
            log.error("Parse failed: {}", e.getMessage(), e);
            throw new CustomException(ErrorCode.FILE_PARSE_FAILED);
        }
    }

    /**
     * RabbitMQ Listener - Parse CV từ Drive.
     *
     * Lưu ý: method này KHÔNG có @Transactional để tránh giữ DB Connection mở
     * trong suốt 75 giây chờ LlamaParse API.
     * Các thao tác ghi DB được đóng gói vào các helper method @Transactional nhỏ.
     */
    @RabbitListener(queues = RabbitMQConfig.CV_UPLOAD_QUEUE, containerFactory = "cvParsingContainerFactory")
    public void parseCV(CVUploadEvent event) {
        int cvId = event.getCvId();
        String tempFilePath = null;

        try {
            // Guard: skip if CV already terminated — happens when a stale requeued message
            // is picked up after the DLQ listener already marked this CV as FAILED.
            CandidateCV currentState = candidateCVRepository.findById(cvId).orElse(null);
            if (currentState == null) {
                log.warn("[PARSE] CV {} not found in DB, discarding message", cvId);
                return;
            }
            if (currentState.getCvStatus() == CVStatus.FAILED) {
                log.warn("[PARSE] CV {} already FAILED, discarding stale requeued message", cvId);
                return;
            }

            // [Transaction 1] Mark PARSING
            markCvAsParsing(cvId);

            // Download file từ Drive về temp (ngoài transaction)
            String fileId = event.getFileId();
            tempFilePath = storageService.downloadFileToTemp(fileId);

            File file = new File(tempFilePath);
            if (!file.exists()) {
                throw new CustomException(ErrorCode.FILE_NOT_FOUND);
            }

            // Gọi LlamaParse API và polling (mất 10-75s) - KHÔNG giữ transaction
            String jobId = uploadFileForCV(tempFilePath);
            String parsedText = pollResult(jobId);

            // Extract information
            String extractedName = extractName(parsedText);
            String extractedEmail = extractEmail(parsedText);

            // [Transaction 2] Persist parse result — status becomes EXTRACTED.
            saveParsedCvResult(cvId, parsedText, extractedName, extractedEmail);

            // Update batch
            processingBatchService.incrementProcessed(event.getBatchId(), true);

            log.info("CV parsed successfully - ID: {} | Name: {} | Email: {}",
                    cvId, extractedName, extractedEmail);

            // Trigger Stage 1 of the extraction pipeline (Gemini metadata call).
            // Re-use CVUploadEvent as the lightweight trigger — ExtractCVListener will
            // load the CV text from the DB and call GeminiExtractionService.
            CVUploadEvent extractTrigger = new CVUploadEvent(cvId, event.getFileId(),
                    event.getPositionId(), event.getBatchId());
            rabbitTemplate.convertAndSend(RabbitMQConfig.CV_EXTRACT_QUEUE, extractTrigger);

        } catch (Exception e) {
            log.error("CV parse failed for cvId {}: {}", cvId, e.getMessage(), e);
            // Re-throw so RabbitMQ routes this message to cv.upload.dlq.
            // CVUploadDlqListener is the single owner of FAILED state + SSE notification.
            throw new RuntimeException("CV parse failed: " + e.getMessage(), e);
        } finally {
            if (tempFilePath != null) {
                storageService.deleteTempFile(tempFilePath);
            }
        }
    }

    /** [T1] Re-fetch CV và đổi status sang EXTRACTING trong 1 transaction. */
    @Transactional
    public void markCvAsParsing(int cvId) {
        CandidateCV cv = candidateCVRepository.findById(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));
        cv.setCvStatus(CVStatus.EXTRACTING);
        candidateCVRepository.save(cv);
    }

    /** [T2] Re-fetch CV và lưu toàn bộ kết quả parse trong 1 transaction. */
    @Transactional
    public void saveParsedCvResult(int cvId, String parsedText, String name, String email) {
        CandidateCV cv = candidateCVRepository.findById(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));
        cv.setCvContent(parsedText);
        if (name != null)
            cv.setName(name);
        if (email != null)
            cv.setEmail(email);
        cv.setCvStatus(CVStatus.EXTRACTED);
        cv.setUpdatedAt(LocalDateTime.now());
        cv.setErrorMessage(null);
        cv.setFailedAt(null);
        candidateCVRepository.save(cv);
    }

    // HELPER METHODS

    /**
     * Upload JD file — forces LlamaParse to produce structured Markdown with
     * consistent headers.
     */
    private String uploadFileForJD(String absolutePath) {
        File file = new File(absolutePath);
        if (!file.exists()) {
            throw new CustomException(ErrorCode.FILE_NOT_FOUND);
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.set("Authorization", "Bearer " + apiKey);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(file));
        body.add("parsing_instruction",
                "You are a professional Document Architect. Extract the full content of this Job Description document and output it as clean, structured Markdown.\n"
                        +
                        "\n" +
                        "CRITICAL RULES — follow all of them strictly:\n" +
                        "\n" +
                        "[RULE 1 — NO MARKDOWN TABLES]\n" +
                        "NEVER use Markdown table syntax (pipes | or dashes ---). " +
                        "If the document contains tables (e.g. Salary, Benefits, Roadmap), convert each row to a plain bullet point in the format '- Key: Value'.\n"
                        +
                        "Example: a table row 'Gross 10-14M | Mentorship 1-on-1' becomes:\n" +
                        "- Salary: Gross 10-14M VND/month\n" +
                        "- Mentorship: 1-on-1 with Senior/Tech Lead weekly\n" +
                        "\n" +
                        "[RULE 2 — PAGE CONTINUITY]\n" +
                        "If content (a bullet list, paragraph, or converted table row) is visually split across two pages, you MUST JOIN them into one unbroken block. "
                        +
                        "Never restart a section in the middle just because the page changed.\n" +
                        "\n" +
                        "[RULE 3 — STRIP NOISE]\n" +
                        "Completely ignore and DELETE: page numbers, page headers, page footers, watermarks, confidentiality notices, "
                        +
                        "division names (e.g. 'TECHNOLOGY DIVISION', 'TALENT ACQUISITION', 'Page 1 of 2', 'Confidential'). "
                        +
                        "These must NOT appear anywhere in the output.\n" +
                        "\n" +
                        "[RULE 4 — HIERARCHY]\n" +
                        "Use '# Job Title' for the document title. Use '## Section Name' for each major section (Overview, Responsibilities, Requirements, Benefits, etc.). "
                        +
                        "Use '-' for bullet points under each section.\n" +
                        "\n" +
                        "[RULE 5 — CLEAN OUTPUT]\n" +
                        "Return ONLY the Markdown content. No code fences, no meta-comments, no apologies, no explanations.");
        body.add("target_pages", "");
        body.add("invalidate_cache", "true");
        // gpt4o_mode disabled: GPT-4o Vision renders each page as a bitmap image
        // independently,
        // which causes tables spanning multiple pages to get refusal errors ("I'm
        // sorry...").
        // LlamaParse's native OCR+LLM pipeline is more reliable for structured text
        // documents.
        body.add("gpt4o_mode", "false");
        body.add("skip_diagonal_text", "true");
        body.add("extract_all_pages", "true");
        body.add("do_not_unroll_columns", "false");
        body.add("page_separator", "false");
        body.add("prefix_or_suffix", "false");
        body.add("continuous_mode", "true");
        body.add("fast_mode", "false");

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

        ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                "https://api.cloud.llamaindex.ai/api/parsing/upload",
                HttpMethod.POST,
                request,
                new ParameterizedTypeReference<Map<String, Object>>() {
                });

        return (String) response.getBody().get("id");
    }

    /**
     * Upload file cho CV - Config chi tiết với parsing instruction đầy đủ
     */
    private String uploadFileForCV(String absolutePath) {
        File file = new File(absolutePath);
        if (!file.exists()) {
            throw new CustomException(ErrorCode.FILE_NOT_FOUND);
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.set("Authorization", "Bearer " + apiKey);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(file));
        body.add("parsing_instruction",
                "Extract all text from this CV/Resume into Markdown exactly. Preserve tables as markdown tables. Extract text and context from informative images or diagrams. Consistently use Markdown headers (#, ##, ###) for standard CV sections. "
                        +
                        "IMPORTANT: Completely ignore and omit all page footers, page headers, page numbers, and confidential watermarks (e.g., 'Page 1', 'Confidential', etc.) from the final output.");
        body.add("result_type", "markdown");
        body.add("target_pages", "");
        body.add("invalidate_cache", "true");
        body.add("gpt4o_mode", "true");
        body.add("skip_diagonal_text", "true");
        body.add("extract_all_pages", "true");
        body.add("do_not_unroll_columns", "false");
        body.add("page_separator", "false");
        body.add("prefix_or_suffix", "false");
        body.add("continuous_mode", "true");
        body.add("fast_mode", "false");

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

        ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                "https://api.cloud.llamaindex.ai/api/parsing/upload",
                HttpMethod.POST,
                request,
                new ParameterizedTypeReference<Map<String, Object>>() {
                });

        return (String) response.getBody().get("id");
    }

    /**
     * Poll kết quả parse từ LlamaParse API.
     *
     * Chiến lược poll:
     * - Mỗi 3 giây check status 1 lần, tối đa 25 lần = tổng cộng tối đa ~75 giây.
     * - Nếu LlamaParse trả về ERROR → ném exception ngay (lỗi thực sự, vào DLQ).
     * - Nếu hết 75 giây vẫn PENDING/PROCESSING → ném RuntimeException (timeout).
     * RabbitMQ sẽ re-queue/retry thông minh theo cấu hình DLX.
     */
    private String pollResult(String jobId) throws InterruptedException {
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + apiKey);
        HttpEntity<Void> request = new HttpEntity<>(headers);

        String statusUrl = "https://api.cloud.llamaindex.ai/api/parsing/job/" + jobId;
        String resultUrl = "https://api.cloud.llamaindex.ai/api/parsing/job/" + jobId + "/result/markdown";

        final int MAX_POLLS = 25; // 25 lần x 3s = 75s tối đa
        final int POLL_INTERVAL_MS = 3000;

        for (int i = 0; i < MAX_POLLS; i++) {
            try {
                ResponseEntity<Map<String, Object>> statusResponse = restTemplate.exchange(
                        statusUrl, HttpMethod.GET, request,
                        new ParameterizedTypeReference<Map<String, Object>>() {
                        });

                Map<String, Object> statusBody = statusResponse.getBody();
                if (statusBody == null) {
                    log.warn("Poll #{} - Empty status body, retrying...", i + 1);
                    Thread.sleep(POLL_INTERVAL_MS);
                    continue;
                }

                String status = (String) statusBody.get("status");
                log.debug("Poll #{}/{} - Status: {} | JobId: {}", i + 1, MAX_POLLS, status, jobId);

                if ("SUCCESS".equals(status)) {
                    ResponseEntity<Map<String, Object>> resultResponse = restTemplate.exchange(
                            resultUrl, HttpMethod.GET, request,
                            new ParameterizedTypeReference<Map<String, Object>>() {
                            });

                    Map<String, Object> resultBody = resultResponse.getBody();
                    if (resultBody != null && resultBody.containsKey("markdown")) {
                        String markdown = (String) resultBody.get("markdown");
                        if (markdown != null) {
                            markdown = markdown.trim();
                            markdown = markdown.replaceAll("^```markdown\\s*", "");
                            markdown = markdown.replaceAll("\\s*```$", "");
                            markdown = sanitizeMarkdown(markdown.trim());
                        }
                        log.info("Parse completed! Text length: {}", markdown.length());
                        return markdown;
                    }
                    // Trả về SUCCESS nhưng không có markdown -> coi như lỗi hard
                    throw new CustomException(ErrorCode.FILE_PARSE_FAILED);

                } else if ("ERROR".equals(status)) {
                    // LlamaParse xác nhận lỗi rõ ràng -> vào DLQ ngay, không retry
                    log.error("LlamaParse reported ERROR for job: {}", jobId);
                    throw new CustomException(ErrorCode.FILE_PARSE_FAILED);
                }
                // PENDING / PROCESSING -> continue polling

            } catch (CustomException e) {
                throw e; // Re-throw lỗi hard, không bọc ngoài
            } catch (Exception e) {
                // Lỗi mạng/timeout khi gọi LlamaParse API -> log và thử lại poll
                log.warn("Poll #{} network error: {}", i + 1, e.getMessage());
            }

            Thread.sleep(POLL_INTERVAL_MS);
        }

        // Hết số lần poll mà chưa xong -> timeout
        // Ném RuntimeException để RabbitMQ xử lý re-queue theo cấu hình DLX
        log.error("LlamaParse polling timed out (75s) for jobId: {}", jobId);
        throw new RuntimeException("LlamaParse parse timeout after " + MAX_POLLS + " polls for job: " + jobId);
    }

    /**
     * Post-processing sanitizer applied to all LlamaParse output — the "last line
     * of defense".
     *
     * Handles two categories of garbage that slip through even with correct instr
     * ctions:
     * 1. AI refusal messages (e.g. "I'm sorry, I can't assist...") emitted when
     * GPT-4o Vision
     * receives an ambiguous or partially-rendered page image.
     * 
     * 2. Orphaned Markdown table separators (|---|---) that appear when a table
     * is partially
     * converted but the instruction is not fully obeyed.
     */

    private String sanitizeMarkdown(String markdown) {
        if (markdown == null || markdown.isBlank()) {
            return markdown;
        }

        // Decode HTML entities injected by LlamaParse's native OCR pipeline.
        // This regression occurs when gpt4o_mode=false: the underlying XML serializer
        // encodes special characters before converting to Markdown output.
        markdown = markdown
                .replace("&#x26;", "&")
                .replace("&amp;", "&")
                .replace("&#x27;", "'")
                .replace("&apos;", "'")
                .replace("&lt;", "<")
                .replace("&#x3C;", "<")
                .replace("&gt;", ">")
                .replace("&#x3E;", ">")
                .replace("&quot;", "\"")
                .replace("&#x22;", "\"")
                .replace("&#x60;", "`")
                .replace("&nbsp;", " ");

        String[] lines = markdown.split("\\r?\\n", -1);
        StringBuilder cleaned = new StringBuilder();
        boolean prevWasBlank = false;

        for (String line : lines) {
            // Strip AI refusal lines (any line that starts with a refusal phrase)
            if (line.matches("(?i)^\\s*i('m|\\s+am)\\s+sorry[,.]?.*") ||
                    line.matches("(?i)^\\s*i\\s+can'?t\\s+(assist|help|process|identify).*") ||
                    line.matches("(?i)^\\s*i\\s+am\\s+unable\\s+to.*") ||
                    line.matches("(?i)^\\s*unfortunately[,.]?\\s+i\\s+(can'?t|am\\s+unable).*")) {
                log.warn("[Sanitize] Stripped AI refusal line: \"{}\"", line.trim());
                continue;
            }

            // Strip orphaned Markdown table separator rows (e.g. |---|---| or just a line
            // of ----)
            if (line.matches("^\\s*[|\\-]+[|\\-\\s]+$")) {
                log.warn("[Sanitize] Stripped orphaned table separator: \"{}\"", line.trim());
                continue;
            }

            // Collapse excessive consecutive blank lines to max 1
            boolean isBlank = line.isBlank();
            if (isBlank && prevWasBlank) {
                continue;
            }
            prevWasBlank = isBlank;

            cleaned.append(line).append("\n");
        }

        return cleaned.toString().trim();
    }

    // Regex extract email

    private String extractEmail(String text) {
        if (text == null || text.isEmpty())
            return null;

        Pattern emailPattern = Pattern.compile(
                "(?i)(?:email|mail)[:\\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})");
        Matcher matcher = emailPattern.matcher(text);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }

        // Fallback – bắt mọi email trong text (phòng khi không có label "Email:")
        matcher = Pattern.compile("([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})").matcher(text);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }

        return null;
    }

    // Regex extract name
    private String extractName(String text) {
        if (text == null || text.isEmpty())
            return null;

        // Loại bỏ ký tự đặc biệt cơ bản trong text
        String cleanedText = text.replaceAll("[#%@\\-!$^&*()_+=\\[\\]{}|\\\\;:\"'<>,/?~`]", " ");

        // Regex chính – tìm các dạng "Name:", "Full Name:", "Họ tên:", "Fullname:"
        Pattern namePattern = Pattern.compile(
                "(?i)(?:^|\\b)(?:name|full name|họ tên)[:\\s]+([A-ZĐ][a-zA-ZĐđ\\s]+?)(?=\\b(?:date|dob|birth|email|phone|address|\\r?\\n|$))");
        Matcher matcher = namePattern.matcher(cleanedText);
        if (matcher.find()) {
            String name = matcher.group(1).trim();

            // Loại bỏ các phần thừa (nếu có)
            name = name.replaceAll("(?i)\\b(date of birth|dob|email|phone|address).*", "").trim();

            // Loại bỏ các ký tự không phải chữ và space
            name = name.replaceAll("[^a-zA-ZĐđ\\s]", "").trim();

            // Giới hạn độ dài hợp lý
            if (name.length() > 50)
                name = name.substring(0, 50).trim();
            return name;
        }

        // Fallback #1: Dòng đầu tiên
        String[] lines = cleanedText.split("\\r?\\n");
        if (lines.length > 0) {
            String firstLine = lines[0].trim();
            if (firstLine.length() <= 40 && !firstLine.matches(".*[@0-9,.:/].*")) {
                // Chỉ giữ chữ và space
                return firstLine.replaceAll("[^a-zA-ZĐđ\\s]", "").trim();
            }
        }

        // Fallback #2: Dòng nào đó có dạng chữ
        Matcher fallbackMatcher = Pattern
                .compile("\\b([A-ZĐ][a-zA-ZĐđ]+\\s+[A-ZĐ][a-zA-ZĐđ]+(?:\\s+[A-ZĐ][a-zA-ZĐđ]+)?)\\b")
                .matcher(cleanedText);
        if (fallbackMatcher.find()) {
            return fallbackMatcher.group(1).replaceAll("[^a-zA-ZĐđ\\s]", "").trim();
        }

        return null;
    }
}
