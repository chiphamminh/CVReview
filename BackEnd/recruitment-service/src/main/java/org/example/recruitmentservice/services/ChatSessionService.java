package org.example.recruitmentservice.services;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.exception.CustomException;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.recruitmentservice.dto.request.CreateSessionRequest;
import org.example.recruitmentservice.dto.request.SaveMessageRequest;
import org.example.recruitmentservice.dto.response.ChatMessageResponse;
import org.example.recruitmentservice.dto.response.ChatSessionResponse;
import org.example.recruitmentservice.models.entity.ChatHistory;
import org.example.recruitmentservice.models.entity.ChatSession;
import org.example.recruitmentservice.repository.ChatHistoryRepository;
import org.example.recruitmentservice.repository.ChatSessionRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Xử lý toàn bộ business logic cho chat session và history.
 * Sliding window (20 messages) được áp dụng ở getHistory() để giới hạn LLM
 * context,
 * trong khi FE vẫn có thể lấy full history qua getFullHistory().
 */
@Service
@RequiredArgsConstructor
public class ChatSessionService {

    private final ChatSessionRepository chatSessionRepository;
    private final ChatHistoryRepository chatHistoryRepository;

    /**
     * Tạo session mới với UUID do BE generate. FE sẽ dùng sessionId này cho mọi
     * request tiếp theo.
     */
    @Transactional
    public ChatSessionResponse createSession(CreateSessionRequest request) {
        LocalDateTime now = LocalDateTime.now();
        ChatSession session = ChatSession.builder()
                .sessionId(UUID.randomUUID().toString())
                .userId(request.getUserId())
                .chatbotType(request.getChatbotType())
                .positionId(request.getPositionId())
                .mode(request.getMode())
                .createdAt(now)
                .lastActiveAt(now)
                .build();

        chatSessionRepository.save(session);
        return toSessionResponse(session);
    }

    /**
     * Lấy N messages gần nhất theo sliding window — dùng để build LLM context.
     * Kết quả được đảo ngược để trả về chronological order (cũ → mới).
     */
    public List<ChatMessageResponse> getHistory(String sessionId, int limit) {
        validateSessionExists(sessionId);

        Pageable pageable = PageRequest.of(0, limit);
        List<ChatHistory> messages = chatHistoryRepository
                .findTopNBySessionIdOrderByCreatedAtDesc(sessionId, pageable);

        // Đảo lại để trả về thứ tự cũ → mới (đúng với LLM context)
        Collections.reverse(messages);
        return messages.stream().map(this::toMessageResponse).collect(Collectors.toList());
    }

    /** Persist một message turn và cập nhật lastActiveAt của session. */
    @Transactional
    public ChatMessageResponse saveMessage(SaveMessageRequest request) {
        ChatSession session = chatSessionRepository.findBySessionId(request.getSessionId())
                .orElseThrow(() -> new CustomException(ErrorCode.SESSION_NOT_FOUND));

        ChatHistory history = ChatHistory.builder()
                .sessionId(request.getSessionId())
                .role(request.getRole())
                .content(request.getContent())
                .functionCall(request.getFunctionCall())
                .createdAt(LocalDateTime.now())
                .build();

        chatHistoryRepository.save(history);

        // Cập nhật timestamp hoạt động của session
        session.setLastActiveAt(LocalDateTime.now());
        chatSessionRepository.save(session);

        return toMessageResponse(history);
    }

    /**
     * Lấy danh sách sessions của user — FE dùng để render sidebar.
     * positionId null = tất cả sessions (Candidate), non-null = lọc theo position (HR sidebar).
     */
    public Page<ChatSessionResponse> getUserSessions(String userId, Integer positionId, Pageable pageable) {
        Page<ChatSession> page = (positionId != null)
                ? chatSessionRepository.findByUserIdAndPositionIdOrderByLastActiveAtDesc(userId, positionId, pageable)
                : chatSessionRepository.findByUserIdOrderByLastActiveAtDesc(userId, pageable);
        return page.map(this::toSessionResponse);
    }

    /**
     * Cursor-based pagination cho FE history panel — infinite scroll lên trên.
     * Trả về {@code limit} messages kể từ cursor {@code beforeId} (exclusive),
     * theo thứ tự chronological (cũ → mới) để FE có thể prepend trực tiếp.
     *
     * @param beforeId null = first page (latest messages), non-null = older messages before this id
     */
    public List<ChatMessageResponse> getHistoryPage(String sessionId, int limit, Long beforeId) {
        validateSessionExists(sessionId);
        Pageable pageable = PageRequest.of(0, limit);
        List<ChatHistory> messages;
        if (beforeId != null) {
            messages = chatHistoryRepository.findBySessionIdAndIdBeforeOrderByIdDesc(sessionId, beforeId, pageable);
        } else {
            messages = chatHistoryRepository.findTopBySessionIdOrderByIdDesc(sessionId, pageable);
        }
        // Đảo lại để trả về chronological order (cũ → mới) cho FE render đúng
        Collections.reverse(messages);
        return messages.stream().map(this::toMessageResponse).collect(Collectors.toList());
    }

    /**
     * Lấy full chat history của 1 session — dùng nội bộ cho LLM context full load.
     */
    public List<ChatMessageResponse> getFullHistory(String sessionId) {
        validateSessionExists(sessionId);
        return chatHistoryRepository.findBySessionIdOrderByCreatedAtAsc(sessionId)
                .stream().map(this::toMessageResponse).collect(Collectors.toList());
    }

    // Helper methods

    private void validateSessionExists(String sessionId) {
        if (!chatSessionRepository.existsById(sessionId)) {
            throw new CustomException(ErrorCode.SESSION_NOT_FOUND);
        }
    }

    private ChatSessionResponse toSessionResponse(ChatSession session) {
        return ChatSessionResponse.builder()
                .sessionId(session.getSessionId())
                .userId(session.getUserId())
                .chatbotType(session.getChatbotType())
                .positionId(session.getPositionId())
                .mode(session.getMode())
                .createdAt(session.getCreatedAt())
                .lastActiveAt(session.getLastActiveAt())
                .build();
    }

    private ChatMessageResponse toMessageResponse(ChatHistory history) {
        return ChatMessageResponse.builder()
                .id(history.getId())
                .sessionId(history.getSessionId())
                .role(history.getRole())
                .content(history.getContent())
                .functionCall(history.getFunctionCall())
                .createdAt(history.getCreatedAt())
                .build();
    }
}
