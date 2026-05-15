package org.example.recruitmentservice.services;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.exception.CustomException;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.recruitmentservice.dto.request.FinalizeApplicationRequest;
import org.example.recruitmentservice.dto.response.FinalizeApplicationResponse;
import org.example.recruitmentservice.models.entity.CVAnalysis;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.enums.MatchStatus;
import org.example.recruitmentservice.models.enums.SourceType;
import org.example.recruitmentservice.repository.CVAnalysisRepository;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.example.recruitmentservice.utils.PositionUtils;

import java.time.LocalDateTime;

/**
 * Xử lý luồng finalize_application từ Candidate chatbot.
 * Business rules:
 * 1. Candidate phải có Master CV (positionId IS NULL).
 * 2. Score phải >= 70 (guardrail bất biến — re-validate dù chatbot đã check).
 * 3. Candidate chưa nộp vào cùng position này (không duplicate Application CV).
 * 4. Application CV được copy từ Master, KHÔNG embed lại Qdrant.
 */
@Service
@RequiredArgsConstructor
public class FinalizeApplicationService {

    private static final int SCORE_THRESHOLD = 70;

    private final CandidateCVRepository candidateCVRepository;
    private final CVAnalysisRepository cvAnalysisRepository;
    private final PositionRepository positionRepository;
    private final org.springframework.web.client.RestTemplate restTemplate;

    @org.springframework.beans.factory.annotation.Value("${EMBEDDING_SERVICE_URL}")
    private String embeddingServiceUrl;

    @Transactional
    public FinalizeApplicationResponse finalizeApplication(FinalizeApplicationRequest request) {
        // 1. Re-validate: must have a passing overallStatus (defense in depth)
        if (request.getOverallStatus() == null ||
                request.getOverallStatus() == MatchStatus.POOR_FIT) {
            throw new CustomException(ErrorCode.SCORE_BELOW_THRESHOLD);
        }
        // technicalScore must exist
        if (request.getTechnicalScore() == null || request.getTechnicalScore() < SCORE_THRESHOLD) {
            throw new CustomException(ErrorCode.SCORE_BELOW_THRESHOLD);
        }

        // 2. Lấy Master CV của candidate
        CandidateCV masterCv = candidateCVRepository
                .findMasterCvByCandidateId(request.getCandidateId())
                .orElseThrow(() -> new CustomException(ErrorCode.MASTER_CV_NOT_FOUND));

        // 3. Kiểm tra candidate chưa nộp vào position này
        checkDuplicateApplication(request.getCandidateId(), request.getPositionId());

        // 4. Lấy position
        Positions position = positionRepository.findById(request.getPositionId())
                .orElseThrow(() -> new CustomException(ErrorCode.POSITION_NOT_FOUND));

        // 5. Copy Master CV → tạo Application CV
        LocalDateTime now = LocalDateTime.now();
        CandidateCV applicationCv = buildApplicationCv(masterCv, position, now);
        candidateCVRepository.save(applicationCv);

        // 6. Tạo CVAnalysis từ chatbot scoring result
        CVAnalysis analysis = buildCvAnalysis(applicationCv, position, request, now);
        cvAnalysisRepository.save(analysis);

        // 7. Sync Phase 3: Update applied_position_ids array on Master CV in Qdrant
        syncAppliedPositionToQdrant(masterCv.getId(), position.getId());

        return FinalizeApplicationResponse.builder()
                .applicationCvId(applicationCv.getId())
                .message("Application submitted successfully")
                .appliedAt(now)
                .build();
    }

    private void syncAppliedPositionToQdrant(Integer masterCvId, Integer positionId) {
        try {
            // masterCvId matches the cvId stored in Qdrant payloads for Master chunks
            String url = String.format("%s/cv/%d/applied-positions/%d",
                    embeddingServiceUrl, masterCvId, positionId);

            // Using POST instead of PATCH to avoid default JDK RestTemplate limitations
            restTemplate.exchange(
                    url,
                    org.springframework.http.HttpMethod.POST,
                    null,
                    String.class);
            System.out.println("[Qdrant Sync] Successfully updated applied_position_ids for Master CV " + masterCvId);
        } catch (Exception e) {
            // We log but don't fail the transaction — sync can be recovered later via
            // script if needed
            System.err.println(
                    "[Qdrant Sync Error] Failed to update Qdrant for Master CV " + masterCvId + ": " + e.getMessage());
        }
    }

    private void checkDuplicateApplication(String candidateId, Integer positionId) {
        boolean alreadyApplied = candidateCVRepository
                .findApplicationsByPositionId(positionId)
                .stream()
                .anyMatch(cv -> candidateId.equals(cv.getCandidateId()));

        if (alreadyApplied) {
            throw new CustomException(ErrorCode.APPLICATION_ALREADY_EXISTS);
        }
    }

    private CandidateCV buildApplicationCv(CandidateCV master, Positions position, LocalDateTime now) {
        CandidateCV app = new CandidateCV();
        app.setCandidateId(master.getCandidateId());
        app.setPosition(position);
        app.setParentCvId(master.getId());
        app.setSourceType(SourceType.EXTERNAL);
        app.setEmail(master.getEmail());
        app.setName(master.getName());
        // Kế thừa Drive file info từ Master — không upload lại
        app.setDriveFileId(master.getDriveFileId());
        app.setDriveFileUrl(master.getDriveFileUrl());
        app.setCvContent(master.getCvContent());
        // Application CV inherits the Drive file and embedding from Master CV.
        // EMBEDDED status reflects that this CV is already indexed in Qdrant via the
        // Master.
        app.setCvStatus(CVStatus.EMBEDDED);
        app.setAppliedDate(now);
        app.setUpdatedAt(now);
        app.setCreatedAt(now);
        return app;
    }

    private CVAnalysis buildCvAnalysis(CandidateCV applicationCv, Positions position,
            FinalizeApplicationRequest request, LocalDateTime now) {
        CVAnalysis analysis = new CVAnalysis();
        analysis.setCandidateCV(applicationCv);
        analysis.setPositionId(position.getId());
        analysis.setPositionName(PositionUtils.formatPositionTitle(position.getSeniority(), position.getTitle()));
        analysis.setTechnicalScore(request.getTechnicalScore());
        analysis.setExperienceScore(request.getExperienceScore());
        analysis.setOverallStatus(request.getOverallStatus());
        analysis.setAiAssessment(request.getAiAssessment());
        analysis.setLearningPath(request.getLearningPath());
        analysis.setAnalysisMethod("CHATBOT");
        return analysis;
    }
}
