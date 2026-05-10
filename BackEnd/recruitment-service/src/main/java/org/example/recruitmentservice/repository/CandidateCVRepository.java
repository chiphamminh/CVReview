package org.example.recruitmentservice.repository;

import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.enums.SourceType;
import org.springframework.data.domain.*;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface CandidateCVRepository extends JpaRepository<CandidateCV, Integer> {
       Page<CandidateCV> findByPositionId(int positionId, Pageable pageable);

       @Query("SELECT c FROM CandidateCV c WHERE c.position.id = :positionId")
       List<CandidateCV> findListCVsByPositionId(@Param("positionId") int positionId);

       Page<CandidateCV> findByPositionIdAndCvStatusIn(int positionId, List<CVStatus> statuses, Pageable pageable);

       List<CandidateCV> findByPositionIdAndCvStatus(int positionId, CVStatus cvStatus);

       List<CandidateCV> findByPositionIdAndCvStatusAndBatchId(int positionId, CVStatus cvStatus, String batchId);

       List<CandidateCV> findByBatchIdAndCvStatus(String batchId, CVStatus cvStatus);

       long countByBatchIdAndCvStatus(String batchId, CVStatus cvStatus);

       int countByPositionId(int positionId);

       @Query("SELECT COUNT(c) FROM CandidateCV c WHERE c.position.id = :positionId AND c.sourceType = :sourceType")
       long countByPositionIdAndSourceType(@Param("positionId") int positionId,
                     @Param("sourceType") SourceType sourceType);

       @Query("SELECT COUNT(c) FROM CandidateCV c WHERE c.updatedAt >= :date")
       long countTotalCVsAfterDate(@Param("date") java.time.LocalDateTime date);

       @Query("SELECT COUNT(c) FROM CandidateCV c WHERE c.cvStatus = :status AND c.updatedAt >= :date")
       long countByCvStatusAndDateAfter(@Param("status") CVStatus status, @Param("date") java.time.LocalDateTime date);

       /**
        * Load CandidateCV cùng Position trong 1 query → tránh
        * LazyInitializationException
        * khi truy cập cv.getPosition() bên ngoài Hibernate session.
        */
       @Query("SELECT c FROM CandidateCV c LEFT JOIN FETCH c.position WHERE c.id = :id")
       Optional<CandidateCV> findByIdWithPosition(@Param("id") int id);

       /**
        * GC Job: Fetch only FAILED CVs that still have a Drive file pending deletion.
        * driveFileId != null ensures we don't attempt deleting already-cleaned
        * records.
        */
       @Query("SELECT c FROM CandidateCV c WHERE c.cvStatus = 'FAILED' AND c.driveFileId IS NOT NULL AND c.deletedAt IS NULL")
       List<CandidateCV> findFailedCVsPendingCleanup();

       @Query(value = "SELECT c FROM CandidateCV c WHERE " +
                     "(:sourceType IS NULL OR c.sourceType = :sourceType) AND " +
                     "(:status IS NULL OR c.cvStatus = :status)")
       Page<org.example.recruitmentservice.dto.response.AdminCvSummaryDto> findAdminCvList(
                     @Param("sourceType") SourceType sourceType,
                     @Param("status") CVStatus status,
                     Pageable pageable);

       // -------------------------------------------------------
       // Queries phục vụ Chatbot feature
       // -------------------------------------------------------

       /**
        * Tìm Master CV của candidate (positionId IS NULL và chưa bị soft-delete).
        * Dùng để kiểm tra duplicate khi upload và lấy nguồn dữ liệu cho
        * finalize_application.
        */
       @Query("SELECT c FROM CandidateCV c WHERE c.candidateId = :candidateId AND c.position IS NULL AND c.deletedAt IS NULL")
       Optional<CandidateCV> findMasterCvByCandidateId(@Param("candidateId") String candidateId);

       /**
        * Tìm tất cả Application CVs của candidate thông qua parentCvId.
        * Dùng khi candidate re-upload: soft-delete toàn bộ application CVs cũ.
        */
       @Query("SELECT c FROM CandidateCV c WHERE c.parentCvId = :parentCvId AND c.deletedAt IS NULL")
       List<CandidateCV> findApplicationsByParentCvId(@Param("parentCvId") int parentCvId);

       /**
        * Lấy các Application CVs ứng tuyển vào 1 position cụ thể
        * (sourceType=CANDIDATE).
        * Dùng trong HR chatbot mode CANDIDATE để lấy candidateIds rồi filter Qdrant.
        */
       @Query("SELECT c FROM CandidateCV c WHERE c.position.id = :positionId AND c.deletedAt IS NULL")
       List<CandidateCV> findApplicationsByPositionId(@Param("positionId") int positionId);

       /**
        * Đếm số ứng viên Candidate đã nộp vào 1 position.
        * Dùng cho HR dashboard hiển thị số applicants.
        */
       @Query("SELECT COUNT(c) FROM CandidateCV c WHERE c.position.id = :positionId AND c.sourceType = 'CANDIDATE' AND c.deletedAt IS NULL")
       long countApplicationsByPositionId(@Param("positionId") int positionId);

       /**
        * Lấy tất cả Application CVs của một candidate — dùng cho
        * check_application_status tool.
        * Chỉ trả về các bản ghi không bị soft-delete và có positionId (Application CV,
        * không phải Master).
        */
       @Query("SELECT c FROM CandidateCV c LEFT JOIN FETCH c.position WHERE c.candidateId = :candidateId AND c.position IS NOT NULL AND c.sourceType = 'CANDIDATE' AND c.deletedAt IS NULL")
       List<CandidateCV> findApplicationsByCandidateId(@Param("candidateId") String candidateId);

       /**
        * Lấy Application CV của candidate cho một position cụ thể.
        * Dùng khi check_application_status tool muốn kiểm tra vị trí cụ thể.
        */
       @Query("SELECT c FROM CandidateCV c LEFT JOIN FETCH c.position WHERE c.candidateId = :candidateId AND c.position.id = :positionId AND c.sourceType = 'CANDIDATE' AND c.deletedAt IS NULL")
       List<CandidateCV> findApplicationsByCandidateIdAndPositionId(
                     @Param("candidateId") String candidateId,
                     @Param("positionId") int positionId);

       @Query("SELECT c FROM CandidateCV c LEFT JOIN FETCH c.position " +
                     "WHERE (:keyword IS NULL OR :keyword = '' OR LOWER(c.name) LIKE LOWER(CONCAT('%', :keyword, '%')) OR LOWER(c.email) LIKE LOWER(CONCAT('%', :keyword, '%'))) "
                     +
                     "AND c.position IS NOT NULL " +
                     "AND (:positionId IS NULL OR c.position.id = :positionId) " +
                     "AND (:stage IS NULL OR c.recruitmentStage = :stage) " +
                     "AND (:sourceType IS NULL OR c.sourceType = :sourceType) " +
                     "AND (:cvStatus IS NULL OR c.cvStatus = :cvStatus) " +
                     "AND c.deletedAt IS NULL")
       Page<CandidateCV> filterCandidates(
                     @Param("keyword") String keyword,
                     @Param("positionId") Integer positionId,
                     @Param("stage") RecruitmentStage stage,
                     @Param("sourceType") SourceType sourceType,
                     @Param("cvStatus") CVStatus cvStatus,
                     Pageable pageable);

       /** Tìm các CV đã qua ngày phỏng vấn mà vẫn đang ở stage INTERVIEW_SCHEDULED. */
       @Query("SELECT c FROM CandidateCV c " +
                     "WHERE c.recruitmentStage = :stage AND c.interviewSchedule <= :now AND c.deletedAt IS NULL")
       List<CandidateCV> findInterviewsPastDue(
                     @Param("stage") RecruitmentStage stage,
                     @Param("now") LocalDateTime now);
}
