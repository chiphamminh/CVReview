package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.models.entity.Positions;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface PositionRepository extends JpaRepository<Positions, Integer> {
        Positions findById(int positionId);

        Optional<Positions> findByTitleAndSeniority(String title, String seniority);

        @Query("SELECT p FROM Positions p " +
                        "WHERE (:title IS NULL OR LOWER(p.title) = LOWER(:title)) " +
                        "AND (:seniority IS NULL OR LOWER(p.seniority) = LOWER(:seniority))")
        List<Positions> findByFilters(@Param("title") String title,
                        @Param("seniority") String seniority);

        @Query("SELECT DISTINCT p FROM Positions p LEFT JOIN p.skills s " +
                        "WHERE LOWER(p.title) LIKE LOWER(CONCAT('%', :keyword, '%')) " +
                        "   OR LOWER(p.seniority) LIKE LOWER(CONCAT('%', :keyword, '%')) " +
                        "   OR LOWER(s) LIKE LOWER(CONCAT('%', :keyword, '%'))")
        List<Positions> searchByKeyword(@Param("keyword") String keyword);

        @Query("SELECT p FROM Positions p JOIN p.candidateCVs c WHERE c.id = :cvId")
        Positions findByCandidateCVId(@Param("cvId") int cvId);

        /**
         * Lấy tất cả positions đang active — dùng bởi chatbot-service để filter scope
         * JD khi search.
         * Strategy: always-fresh (gọi SQL mỗi request) thay vì sync vào Qdrant
         * metadata.
         */
        @Query("SELECT p FROM Positions p WHERE p.isActive = true ORDER BY p.openedAt DESC")
        List<Positions> findAllActive();

        long countByBatchIdAndStatus(String batchId, org.example.recruitmentservice.models.enums.JDStatus status);

        List<Positions> findByBatchIdAndStatus(String batchId,
                        org.example.recruitmentservice.models.enums.JDStatus status);
}
