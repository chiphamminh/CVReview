package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.dto.response.ScoreTrendProjection;
import org.example.recruitmentservice.models.entity.CVAnalysis;
import org.example.recruitmentservice.models.enums.SourceType;
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

        // -------------------------------------------------------
        // Queries phục vụ HR Analytics Dashboard (position-filtered)
        // -------------------------------------------------------

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND a.candidateCV.updatedAt >= :dateSince")
        long countScoredByPositionAfterDate(@Param("positionId") int positionId, @Param("dateSince") LocalDateTime dateSince);

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND (a.technicalScore + a.experienceScore) >= :minSum " +
                        "AND a.candidateCV.updatedAt >= :dateSince")
        long countPassedByPositionAfterDate(@Param("positionId") int positionId, @Param("minSum") int minSum, @Param("dateSince") LocalDateTime dateSince);

        @Query("SELECT AVG((a.technicalScore + a.experienceScore) / 2.0) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND a.candidateCV.updatedAt >= :dateSince")
        Double averageScoreByPositionAfterDate(@Param("positionId") int positionId, @Param("dateSince") LocalDateTime dateSince);

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND (a.technicalScore + a.experienceScore) >= :minSum " +
                        "AND (a.technicalScore + a.experienceScore) < :maxSum")
        long countInRangeByPosition(@Param("positionId") int positionId, @Param("minSum") int minSum, @Param("maxSum") int maxSum);

        /** Avg score tất cả thời gian cho 1 position — dùng cho positions health table. */
        @Query("SELECT AVG((a.technicalScore + a.experienceScore) / 2.0) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND a.candidateCV.deletedAt IS NULL")
        Double averageScoreByPosition(@Param("positionId") int positionId);

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND (a.technicalScore + a.experienceScore) >= :minSum " +
                        "AND a.candidateCV.deletedAt IS NULL")
        long countQualifiedByPosition(@Param("positionId") int positionId, @Param("minSum") int minSum);

        // Source breakdown — all positions
        @Query("SELECT AVG((a.technicalScore + a.experienceScore) / 2.0) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.sourceType = :sourceType " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND a.candidateCV.updatedAt >= :dateSince " +
                        "AND a.candidateCV.position IS NOT NULL")
        Double avgScoreBySourceAfterDate(@Param("sourceType") SourceType sourceType, @Param("dateSince") LocalDateTime dateSince);

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.sourceType = :sourceType " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND a.candidateCV.updatedAt >= :dateSince " +
                        "AND a.candidateCV.position IS NOT NULL")
        long countScoredBySourceAfterDate(@Param("sourceType") SourceType sourceType, @Param("dateSince") LocalDateTime dateSince);

        /** Threshold hardcode 140 (composite ≥ 70) cho all-positions source breakdown. */
        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.sourceType = :sourceType " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND a.candidateCV.updatedAt >= :dateSince " +
                        "AND a.candidateCV.position IS NOT NULL " +
                        "AND (a.technicalScore + a.experienceScore) >= 140")
        long countPassedBySourceAfterDate(@Param("sourceType") SourceType sourceType, @Param("dateSince") LocalDateTime dateSince);

        // Source breakdown — specific position
        @Query("SELECT AVG((a.technicalScore + a.experienceScore) / 2.0) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.candidateCV.sourceType = :sourceType " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL")
        Double avgScoreByPositionAndSource(@Param("positionId") int positionId, @Param("sourceType") SourceType sourceType);

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.candidateCV.sourceType = :sourceType " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL")
        long countScoredByPositionAndSource(@Param("positionId") int positionId, @Param("sourceType") SourceType sourceType);

        @Query("SELECT COUNT(a) FROM CVAnalysis a " +
                        "WHERE a.candidateCV.position.id = :positionId " +
                        "AND a.candidateCV.sourceType = :sourceType " +
                        "AND a.technicalScore IS NOT NULL AND a.experienceScore IS NOT NULL " +
                        "AND (a.technicalScore + a.experienceScore) >= :minSum")
        long countPassedByPositionAndSource(@Param("positionId") int positionId, @Param("sourceType") SourceType sourceType, @Param("minSum") int minSum);

        // Score trend — native SQL, GROUP BY week
        @Query(value = "SELECT YEAR(cv.created_at) as yr, WEEK(cv.created_at) as wk, " +
                "ROUND(AVG((a.technical_score + a.experience_score) / 2.0), 1) as avgScore, " +
                "COUNT(a.id) as cvCount " +
                "FROM candidate_cv cv " +
                "JOIN cv_analysis a ON a.cv_id = cv.id " +
                "WHERE cv.created_at >= :dateSince " +
                "AND a.technical_score IS NOT NULL AND a.experience_score IS NOT NULL " +
                "AND cv.deleted_at IS NULL " +
                "GROUP BY YEAR(cv.created_at), WEEK(cv.created_at) " +
                "ORDER BY yr, wk", nativeQuery = true)
        List<ScoreTrendProjection> getScoreTrendAllPositions(@Param("dateSince") LocalDateTime dateSince);

        @Query(value = "SELECT YEAR(cv.created_at) as yr, WEEK(cv.created_at) as wk, " +
                "ROUND(AVG((a.technical_score + a.experience_score) / 2.0), 1) as avgScore, " +
                "COUNT(a.id) as cvCount " +
                "FROM candidate_cv cv " +
                "JOIN cv_analysis a ON a.cv_id = cv.id " +
                "WHERE cv.position_id = :positionId " +
                "AND cv.created_at >= :dateSince " +
                "AND a.technical_score IS NOT NULL AND a.experience_score IS NOT NULL " +
                "AND cv.deleted_at IS NULL " +
                "GROUP BY YEAR(cv.created_at), WEEK(cv.created_at) " +
                "ORDER BY yr, wk", nativeQuery = true)
        List<ScoreTrendProjection> getScoreTrendByPosition(@Param("positionId") int positionId, @Param("dateSince") LocalDateTime dateSince);
}
