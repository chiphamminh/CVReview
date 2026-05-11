package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.models.entity.CVAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface CVAnalysisRepository extends JpaRepository<CVAnalysis, Integer> {

        Optional<CVAnalysis> findByCandidateCV_Id(Integer cvId);

        /**
         * Đếm số CVAnalysis records cho một vị trí và nguồn cụ thể.
         */
        @Query("SELECT COUNT(a) FROM CVAnalysis a WHERE a.candidateCV.position.id = :positionId AND a.candidateCV.sourceType = :sourceType")
        long countScoredByPositionIdAndSourceType(@Param("positionId") int positionId,
                        @Param("sourceType") org.example.recruitmentservice.models.enums.SourceType sourceType);

        /** Đếm số CV đạt ngưỡng điểm pass cho một vị trí và nguồn cụ thể. */
        @Query("SELECT COUNT(a) FROM CVAnalysis a WHERE a.candidateCV.position.id = :positionId AND a.candidateCV.sourceType = :sourceType AND a.technicalScore >= :threshold")
        long countPassedByPositionIdAndSourceType(@Param("positionId") int positionId,
                        @Param("sourceType") org.example.recruitmentservice.models.enums.SourceType sourceType,
                        @Param("threshold") int threshold);

        /** Lấy toàn bộ CVAnalysis của một list các CV ID để tránh N+1. */
        List<CVAnalysis> findByCandidateCV_IdIn(List<Integer> cvIds);

        // -------------------------------------------------------
        // Queries phục vụ HR Analytics Dashboard
        // -------------------------------------------------------

        /** Số CV đã được chấm điểm (cả 2 score không null) trong khoảng thời gian. */
        @Query("SELECT COUNT(a) FROM CVAnalysis a WHERE a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL AND a.candidateCV.updatedAt >= :dateSince")
        long countScoredAfterDate(@Param("dateSince") LocalDateTime dateSince);

        /** Số CV có composite score >= 70 (tức sum >= 140) trong khoảng thời gian. */
        @Query("SELECT COUNT(a) FROM CVAnalysis a WHERE a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL AND (a.technicalScore + a.experienceScore) >= 140 AND a.candidateCV.updatedAt >= :dateSince")
        long countPassedAfterDate(@Param("dateSince") LocalDateTime dateSince);

        /** Điểm composite trung bình trong khoảng thời gian. */
        @Query("SELECT AVG((a.technicalScore + a.experienceScore) / 2.0) FROM CVAnalysis a WHERE a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL AND a.candidateCV.updatedAt >= :dateSince")
        Double averageCompositeScoreAfterDate(@Param("dateSince") LocalDateTime dateSince);

        /**
         * Đếm CV trong dải composite score [minSum/2, maxSum/2).
         * Dùng sum thay vì chia để tránh floating-point trong JPQL.
         */
        @Query("SELECT COUNT(a) FROM CVAnalysis a WHERE a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL AND (a.technicalScore + a.experienceScore) >= :minSum AND (a.technicalScore + a.experienceScore) < :maxSum")
        long countInCompositeRange(@Param("minSum") int minSum, @Param("maxSum") int maxSum);
}
