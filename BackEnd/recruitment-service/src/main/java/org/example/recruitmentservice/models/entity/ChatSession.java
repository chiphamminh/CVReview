package org.example.recruitmentservice.models.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.example.recruitmentservice.models.enums.ChatMode;
import org.example.recruitmentservice.models.enums.ChatbotType;

import java.time.LocalDateTime;

/**
 * Đại diện cho một phiên chat (session). Mỗi lần user bấm "New Chat" → 1
 * session mới.
 * Session_id được BE tạo (UUID) và trả về FE để FE đính kèm vào mọi request
 * sau.
 */
@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "chat_session")
public class ChatSession {

    @Id
    @Column(name = "session_id", length = 36, nullable = false)
    private String sessionId;

    /** ID của HR hoặc Candidate đang chat. */
    @Column(name = "user_id", nullable = false)
    private String userId;

    @Column(name = "chatbot_type", nullable = false)
    @Enumerated(EnumType.STRING)
    private ChatbotType chatbotType;

    /**
     * positionId xác định phạm vi dữ liệu của HR chatbot.
     * NULL đối với Candidate chatbot.
     */
    @Column(name = "position_id")
    private Integer positionId;

    /**
     * Mode làm việc của HR chatbot: INTERNAL (CV từ HR upload) hoặc EXTERNAL (CV từ
     * Candidate nộp).
     * NULL đối với Candidate chatbot.
     */
    @Column(name = "mode")
    @Enumerated(EnumType.STRING)
    private ChatMode mode;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    /**
     * Cập nhật mỗi khi có message mới — dùng để sắp xếp danh sách sessions cho FE.
     */
    @Column(name = "last_active_at", nullable = false)
    private LocalDateTime lastActiveAt;
}
