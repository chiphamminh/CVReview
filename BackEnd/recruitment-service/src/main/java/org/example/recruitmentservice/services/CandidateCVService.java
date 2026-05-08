package org.example.recruitmentservice.services;

import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;
import com.google.api.client.util.Value;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.dto.response.PageResponse;
import org.example.commonlibrary.exception.CustomException;
import org.example.commonlibrary.utils.PageUtil;
import org.example.recruitmentservice.dto.response.CandidateCVResponse;
import org.example.recruitmentservice.models.entity.CVAnalysis;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.repository.CVAnalysisRepository;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class CandidateCVService {
    private final CandidateCVRepository candidateCVRepository;
    private final PositionRepository positionRepository;
    private final CVAnalysisRepository cvAnalysisRepository;
    private final StorageService storageService;
    private final RestTemplate restTemplate;

    @Value("${EMBEDDING_SERVICE_URL}")
    private String embeddingServiceUrl;

    public ApiResponse<CandidateCVResponse> getCVDetail(int cvId) {
        CandidateCV cv = candidateCVRepository.findById(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        Optional<CVAnalysis> cvAnalysisOpt = cvAnalysisRepository.findByCandidateCV_Id(cvId);
        CVAnalysis cvAnalysis = cvAnalysisOpt.orElse(null);

        CandidateCVResponse response = CandidateCVResponse.builder()
                .cvId(cv.getId())
                .positionId(cv.getPosition().getId())
                .positionTitle(cv.getPosition().getTitle())
                .name(cv.getName())
                .email(cv.getEmail())
                .batchId(cv.getBatchId())
                .driveFileUrl(cv.getDriveFileUrl())
                .technicalScore(cvAnalysis != null ? cvAnalysis.getTechnicalScore() : null)
                .experienceScore(cvAnalysis != null ? cvAnalysis.getExperienceScore() : null)
                .overallStatus(cvAnalysis != null ? cvAnalysis.getOverallStatus() : null)
                .aiAssessment(cvAnalysis != null ? cvAnalysis.getAiAssessment() : null)
                .learningPath(cvAnalysis != null ? cvAnalysis.getLearningPath() : null)
                .status(cv.getCvStatus())
                .errorMessage(cv.getErrorMessage())
                .failedAt(cv.getFailedAt())
                .retryCount(cv.getRetryCount())
                .canRetry(cv.getCvStatus() == CVStatus.FAILED)
                .createdAt(cv.getCreatedAt())
                .recruitmentStage(cv.getRecruitmentStage())
                .appliedDate(cv.getAppliedDate())
                .interviewSchedule(cv.getInterviewSchedule())
                .build();

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "CV detail retrieved successfully",
                response);
    }

    public ApiResponse<PageResponse<CandidateCVResponse>> getAllCVsByPositionId(Integer positionId,
            List<CVStatus> statuses, int page, int size) {
        Positions position = positionRepository.findById(positionId)
                .orElseThrow(() -> new CustomException(ErrorCode.POSITION_NOT_FOUND));

        if (statuses == null || statuses.isEmpty()) {
            statuses = List.of(CVStatus.EXTRACTED, CVStatus.EMBEDDED, CVStatus.FAILED);
        }

        Pageable pageable = PageRequest.of(page, size, Sort.by("updatedAt").descending());
        Page<CandidateCV> cvPage = candidateCVRepository.findByPositionIdAndCvStatusIn(positionId, statuses, pageable);

        Page<CandidateCVResponse> mappedPage = cvPage.map(cv -> CandidateCVResponse.builder()
                .cvId(cv.getId())
                .positionId(cv.getPosition().getId())
                .positionTitle(cv.getPosition().getTitle())
                .batchId(cv.getBatchId())
                .status(cv.getCvStatus())
                .name(cv.getName())
                .email(cv.getEmail())
                .updatedAt(cv.getUpdatedAt())
                .driveFileUrl(cv.getDriveFileUrl())
                .build());

        return ApiResponse.<PageResponse<CandidateCVResponse>>builder()
                .statusCode(ErrorCode.SUCCESS.getCode())
                .message("Fetched all CVs for position: " + position.getTitle())
                .data(PageUtil.toPageResponse(mappedPage))
                .timestamp(LocalDateTime.now())
                .build();
    }

    /**
     * Update thông tin cơ bản của CV (name, email).
     * Không cho phép thay đổi file — CV đã nộp là một snapshot cố định.
     */
    @Transactional
    public void updateCV(int cvId, String name, String email) {
        CandidateCV cv = candidateCVRepository.findById(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        if (name != null && !name.trim().isEmpty()) {
            cv.setName(name.trim());
        }
        if (email != null && !email.trim().isEmpty()) {
            cv.setEmail(email.trim());
        }

        cv.setUpdatedAt(LocalDateTime.now());
        candidateCVRepository.save(cv);
    }

    @Transactional
    public void deleteCandidateCVs(List<Integer> cvIds) {

        for (Integer cvId : cvIds) {

            CandidateCV cv = candidateCVRepository.findById(cvId)
                    .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

            // Xóa embeddings trên Python service
            try {
                String url = embeddingServiceUrl + "/cv/" + cvId;
                restTemplate.delete(url);
                System.out.println("Deleted embeddings for candidate CV: " + cvId);
            } catch (Exception e) {
                System.err.println("Failed to delete embeddings for candidate CV " + cvId + ": " + e.getMessage());
                // Continue anyway - không block việc xóa candidate CV
            }

            // Xóa file trên GG Drive
            try {
                if (cv.getDriveFileId() != null) {
                    storageService.deleteFile(cv.getDriveFileId());
                }
            } catch (Exception e) {
                throw new CustomException(ErrorCode.FILE_DELETE_FAILED);
            }

            candidateCVRepository.delete(cv);
        }
    }
}
