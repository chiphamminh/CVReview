package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.models.entity.ChatHistory;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface ChatHistoryRepository extends JpaRepository<ChatHistory, Long> {

    /**
     * Lấy N messages gần nhất của session (sliding window cho LLM context).
     * Dùng ORDER BY DESC + LIMIT để chỉ lấy 20 rows, sau đó đảo lại ở service layer.
     */
    @Query("SELECT h FROM ChatHistory h WHERE h.sessionId = :sessionId ORDER BY h.createdAt DESC")
    List<ChatHistory> findTopNBySessionIdOrderByCreatedAtDesc(
            @Param("sessionId") String sessionId,
            Pageable pageable);

    /** Cursor pagination — first page (no cursor): lấy N messages mới nhất theo id DESC. */
    @Query("SELECT h FROM ChatHistory h WHERE h.sessionId = :sessionId ORDER BY h.id DESC")
    List<ChatHistory> findTopBySessionIdOrderByIdDesc(
            @Param("sessionId") String sessionId,
            Pageable pageable);

    /** Cursor pagination — with cursor: lấy N messages có id < beforeId theo id DESC. */
    @Query("SELECT h FROM ChatHistory h WHERE h.sessionId = :sessionId AND h.id < :beforeId ORDER BY h.id DESC")
    List<ChatHistory> findBySessionIdAndIdBeforeOrderByIdDesc(
            @Param("sessionId") String sessionId,
            @Param("beforeId") Long beforeId,
            Pageable pageable);

    /** Lấy toàn bộ history của session theo thứ tự thời gian — dùng cho LLM context full load. */
    List<ChatHistory> findBySessionIdOrderByCreatedAtAsc(String sessionId);
}
