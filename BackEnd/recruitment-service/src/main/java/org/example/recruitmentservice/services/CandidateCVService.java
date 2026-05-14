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
import org.example.recruitmentservice.utils.PositionUtils;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.models.enums.SourceType;
import org.example.recruitmentservice.models.enums.EmailType;
import org.example.recruitmentservice.dto.request.InterviewNotificationRequest;
import org.springframework.web.multipart.MultipartFile;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
@RequiredArgsConstructor
public class CandidateCVService {
    private final CandidateCVRepository candidateCVRepository;
    private final PositionRepository positionRepository;
    private final CVAnalysisRepository cvAnalysisRepository;
    private final StorageService storageService;
    private final RestTemplate restTemplate;
    private final NotificationService notificationService;

    @Value("${EMBEDDING_SERVICE_URL}")
    private String embeddingServiceUrl;

    public ApiResponse<List<CandidateCVResponse>> getMyApplications(String candidateId) {
        List<CandidateCV> applications = candidateCVRepository.findApplicationsByCandidateId(candidateId);

        List<CandidateCVResponse> responseList = applications.stream().map(cv -> {
            CVAnalysis cvAnalysis = cv.getAnalysis();
            return CandidateCVResponse.builder()
                    .cvId(cv.getId())
                    .positionId(cv.getPosition() != null ? cv.getPosition().getId() : 0)
                    .positionTitle(
                            cv.getPosition() != null
                                    ? PositionUtils.formatPositionTitle(cv.getPosition().getSeniority(),
                                            cv.getPosition().getTitle())
                                    : null)
                    .technicalScore(cvAnalysis != null ? cvAnalysis.getTechnicalScore() : null)
                    .experienceScore(cvAnalysis != null ? cvAnalysis.getExperienceScore() : null)
                    .overallStatus(cvAnalysis != null ? cvAnalysis.getOverallStatus() : null)
                    .aiAssessment(cvAnalysis != null ? cvAnalysis.getAiAssessment() : null)
                    .learningPath(cvAnalysis != null ? cvAnalysis.getLearningPath() : null)
                    .recruitmentStage(cv.getRecruitmentStage())
                    .interviewSchedule(cv.getInterviewSchedule())
                    .appliedDate(cv.getAppliedDate())
                    .build();
        }).toList();

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Applications retrieved successfully",
                responseList);
    }

    public ApiResponse<CandidateCVResponse> getMasterCV(String candidateId) {
        CandidateCV cv = candidateCVRepository.findMasterCvByCandidateId(candidateId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        CandidateCVResponse response = CandidateCVResponse.builder()
                .cvId(cv.getId())
                .name(cv.getName())
                .email(cv.getEmail())
                .driveFileUrl(cv.getDriveFileUrl())
                .status(cv.getCvStatus())
                .sourceType(cv.getSourceType())
                .createdAt(cv.getCreatedAt())
                .updatedAt(cv.getUpdatedAt())
                .build();

        return new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Master CV retrieved successfully",
                response);
    }

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
                .sourceType(cv.getSourceType())
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

    public ApiResponse<PageResponse<CandidateCVResponse>> filterCandidates(String keyword, Integer positionId,
            RecruitmentStage stage, SourceType sourceType, CVStatus cvStatus, Boolean isScored, String scoreSort,
            int page, int size) {
        // JPQL handles the sorting logic natively, so we pass an unsorted Pageable
        Pageable pageable = PageRequest.of(page, size);
        Page<CandidateCV> cvPage = candidateCVRepository.filterCandidates(keyword, positionId, stage, sourceType,
                cvStatus, isScored, scoreSort, pageable);

        Page<CandidateCVResponse> mappedPage = cvPage.map(cv -> {
            CVAnalysis cvAnalysis = cv.getAnalysis();
            return CandidateCVResponse.builder()
                    .cvId(cv.getId())
                    .positionId(cv.getPosition() != null ? cv.getPosition().getId() : 0)
                    .positionTitle(cv.getPosition() != null
                            ? PositionUtils.formatPositionTitle(cv.getPosition().getSeniority(),
                                    cv.getPosition().getTitle())
                            : null)
                    .email(cv.getEmail())
                    .name(cv.getName())
                    .batchId(cv.getBatchId())
                    .driveFileUrl(cv.getDriveFileUrl())
                    .technicalScore(cvAnalysis != null ? cvAnalysis.getTechnicalScore() : null)
                    .experienceScore(cvAnalysis != null ? cvAnalysis.getExperienceScore() : null)
                    .overallStatus(cvAnalysis != null ? cvAnalysis.getOverallStatus() : null)
                    .aiAssessment(cvAnalysis != null ? cvAnalysis.getAiAssessment() : null)
                    .learningPath(cvAnalysis != null ? cvAnalysis.getLearningPath() : null)
                    .status(cv.getCvStatus())
                    .sourceType(cv.getSourceType())
                    .recruitmentStage(cv.getRecruitmentStage())
                    .errorMessage(cv.getErrorMessage())
                    .failedAt(cv.getFailedAt())
                    .retryCount(cv.getRetryCount())
                    .createdAt(cv.getCreatedAt())
                    .updatedAt(cv.getUpdatedAt())
                    .appliedDate(cv.getAppliedDate())
                    .interviewSchedule(cv.getInterviewSchedule())
                    .build();
        });

        return ApiResponse.<PageResponse<CandidateCVResponse>>builder()
                .statusCode(ErrorCode.SUCCESS.getCode())
                .message("Candidates filtered successfully")
                .data(PageUtil.toPageResponse(mappedPage))
                .timestamp(LocalDateTime.now())
                .build();
    }

    @Transactional
    public void scheduleInterview(int cvId, LocalDateTime interviewDate, String customMessage) {
        CandidateCV cv = candidateCVRepository.findByIdWithPosition(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        if (cv.getRecruitmentStage() != RecruitmentStage.APPLIED) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION,
                    "Interview can only be scheduled for candidates in APPLIED stage.");
        }

        cv.setRecruitmentStage(RecruitmentStage.INTERVIEW_SCHEDULED);
        cv.setInterviewSchedule(interviewDate);
        cv.setUpdatedAt(LocalDateTime.now());
        candidateCVRepository.save(cv);

        InterviewNotificationRequest request = InterviewNotificationRequest.builder()
                .appCvId(cv.getId())
                .candidateId(cv.getCandidateId())
                .candidateEmail(cv.getEmail())
                .candidateName(cv.getName())
                .positionId(cv.getPosition() != null ? cv.getPosition().getId() : null)
                .positionName(cv.getPosition() != null ? cv.getPosition().getTitle() : "N/A")
                .emailType(EmailType.INTERVIEW_INVITE.name())
                .interviewDate(interviewDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")))
                .customMessage(customMessage)
                .build();

        try {
            notificationService.sendInterviewNotification(request);
        } catch (Exception e) {
            log.error("Failed to send interview invitation email for CV {}: {}", cvId, e.getMessage());
            throw new CustomException(ErrorCode.EMAIL_SEND_FAILED,
                    "Failed to send interview invitation email: " + e.getMessage());
        }
    }

    @Transactional
    public void rescheduleInterview(int cvId, LocalDateTime interviewDate, String customMessage) {
        CandidateCV cv = candidateCVRepository.findByIdWithPosition(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        if (cv.getRecruitmentStage() != RecruitmentStage.INTERVIEW_SCHEDULED) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION,
                    "Interview can only be rescheduled for candidates in INTERVIEW_SCHEDULED stage.");
        }

        cv.setInterviewSchedule(interviewDate);
        cv.setUpdatedAt(LocalDateTime.now());
        candidateCVRepository.save(cv);

        InterviewNotificationRequest request = InterviewNotificationRequest.builder()
                .appCvId(cv.getId())
                .candidateId(cv.getCandidateId())
                .candidateEmail(cv.getEmail())
                .candidateName(cv.getName())
                .positionId(cv.getPosition() != null ? cv.getPosition().getId() : null)
                .positionName(cv.getPosition() != null ? cv.getPosition().getTitle() : "N/A")
                .emailType(EmailType.INTERVIEW_INVITE.name())
                .interviewDate(interviewDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")))
                .customMessage(customMessage)
                .build();

        try {
            notificationService.sendInterviewNotification(request);
        } catch (Exception e) {
            log.error("Failed to send interview reschedule email for CV {}: {}", cvId, e.getMessage());
            throw new CustomException(ErrorCode.EMAIL_SEND_FAILED,
                    "Failed to send interview reschedule email: " + e.getMessage());
        }
    }

    @Transactional
    public void sendOffer(int cvId, String startDate, String offerExpirationDate,
            List<MultipartFile> attachments) {
        CandidateCV cv = candidateCVRepository.findByIdWithPosition(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        if (cv.getRecruitmentStage() != RecruitmentStage.INTERVIEWED) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION,
                    "Offer can only be sent to candidates in INTERVIEWED stage.");
        }

        cv.setRecruitmentStage(RecruitmentStage.OFFER);
        cv.setUpdatedAt(LocalDateTime.now());
        candidateCVRepository.save(cv);

        InterviewNotificationRequest request = InterviewNotificationRequest.builder()
                .appCvId(cv.getId())
                .candidateId(cv.getCandidateId())
                .candidateEmail(cv.getEmail())
                .candidateName(cv.getName())
                .positionId(cv.getPosition() != null ? cv.getPosition().getId() : null)
                .positionName(cv.getPosition() != null ? cv.getPosition().getTitle() : "N/A")
                .emailType(EmailType.OFFER_LETTER.name())
                .startDate(startDate)
                .offerExpirationDate(offerExpirationDate)
                .build();

        try {
            notificationService.sendInterviewNotification(request, attachments);
        } catch (Exception e) {
            log.error("Failed to send offer email for CV {}: {}", cvId, e.getMessage());
            throw new CustomException(ErrorCode.EMAIL_SEND_FAILED, "Failed to send offer email: " + e.getMessage());
        }
    }

    @Transactional
    public void updateStage(int cvId, RecruitmentStage newStage) {
        CandidateCV cv = candidateCVRepository.findById(cvId)
                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

        if (newStage != RecruitmentStage.ACCEPTED && newStage != RecruitmentStage.REJECTED) {
            throw new CustomException(ErrorCode.UNAUTHORIZED_ACTION,
                    "Only ACCEPTED or REJECTED stage updates are allowed manually.");
        }

        cv.setRecruitmentStage(newStage);
        cv.setUpdatedAt(LocalDateTime.now());
        candidateCVRepository.save(cv);
    }
}
