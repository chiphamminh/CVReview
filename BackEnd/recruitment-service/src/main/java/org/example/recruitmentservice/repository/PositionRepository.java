package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.models.entity.Positions;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface PositionRepository extends JpaRepository<Positions, Integer> {
        Positions findById(int positionId);

        Optional<Positions> findByTitleAndSeniority(String title, String seniority);

        @Query("SELECT p FROM Positions p " +
                        "WHERE (:keyword IS NULL OR :keyword = '' " +
                        "       OR LOWER(p.title) LIKE LOWER(CONCAT('%', :keyword, '%')) " +
                        "       OR LOWER(p.seniority) LIKE LOWER(CONCAT('%', :keyword, '%'))) " +
                        "AND (:isActive IS NULL OR p.isActive = :isActive)")
        Page<Positions> filterPositions(
                        @Param("keyword") String keyword,
                        @Param("isActive") Boolean isActive,
                        Pageable pageable);

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
