package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.models.entity.ChatSession;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ChatSessionRepository extends JpaRepository<ChatSession, String> {

    /** Lấy tất cả sessions của user, mới nhất lên đầu. */
    Page<ChatSession> findByUserIdOrderByLastActiveAtDesc(String userId, Pageable pageable);

    /** Lấy sessions của user theo positionId cụ thể — HR sidebar lọc đúng position. */
    Page<ChatSession> findByUserIdAndPositionIdOrderByLastActiveAtDesc(String userId, Integer positionId, Pageable pageable);

    Optional<ChatSession> findBySessionId(String sessionId);
}
