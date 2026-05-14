package org.example.recruitmentservice.services;

import lombok.RequiredArgsConstructor;
import org.example.recruitmentservice.dto.response.ActivePositionResponse;
import org.example.recruitmentservice.dto.response.ApplicationSummaryResponse;
import org.example.recruitmentservice.dto.response.CandidateApplicationStatusResponse;
import org.example.recruitmentservice.dto.response.CvStatisticsResponse;
import org.example.recruitmentservice.dto.response.PositionDetailsResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.exception.CustomException;
import org.example.recruitmentservice.dto.request.EvaluateApplicationRequest;
import org.example.recruitmentservice.models.entity.CVAnalysis;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.models.enums.SourceType;
import org.example.recruitmentservice.repository.CVAnalysisRepository;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Service xử lý các yêu cầu lấy dữ liệu nội bộ (internal) phục vụ cho Chatbot.
 * Tách biệt logic truy xuất dữ liệu và mapping khỏi Controller để tuân thủ SRP.
 */
@Service
@RequiredArgsConstructor
public class ChatbotInternalService {

        private final PositionRepository positionRepository;
        private final CandidateCVRepository candidateCVRepository;
        private final CVAnalysisRepository cvAnalysisRepository;

        /**
         * Returns full JD text for a set of position IDs.
         * Used by chatbot-service for Small-to-Big retrieval:
         * Qdrant returns chunk hits → extract unique positionIds → call this method →
         * pass full JD to Gemini Pro.
         *
         * @param positionIds list of position IDs to retrieve (duplicates are
         *                    de-duplicated)
         */
        public List<PositionDetailsResponse> getPositionDetails(List<Integer> positionIds) {
                if (positionIds == null || positionIds.isEmpty()) {
                        return List.of();
                }
                Set<Integer> uniqueIds = Set.copyOf(positionIds);
                return positionRepository.findAllById(uniqueIds)
                                .stream()
                                .map(this::toPositionDetailsResponse)
                                .collect(Collectors.toList());
        }

        /**
         * Lấy danh sách các vị trí đang mở (active).
         * Dùng cho Chatbot để lọc phạm vi tìm kiếm JD trên Qdrant.
         */
        public List<ActivePositionResponse> getActivePositions() {
                return positionRepository.findAllActive()
                                .stream()
                                .map(this::toActivePositionResponse)
                                .collect(Collectors.toList());
        }

        /**
         * Lấy danh sách tóm tắt các ứng viên đã nộp đơn cho một vị trí cụ thể.
         * HR Chatbot dùng candidateId list này để lọc các Master CV vectors trên
         * Qdrant.
         */
        public List<ApplicationSummaryResponse> getApplicationsByPosition(int positionId) {
                List<CandidateCV> cvs = candidateCVRepository.findApplicationsByPositionId(positionId);
                if (cvs.isEmpty())
                        return List.of();

                List<Integer> cvIds = cvs.stream().map(CandidateCV::getId).collect(Collectors.toList());
                java.util.Map<Integer, org.example.recruitmentservice.models.entity.CVAnalysis> analysisMap = cvAnalysisRepository
                                .findByCandidateCV_IdIn(cvIds).stream()
                                .collect(Collectors.toMap(a -> a.getCandidateCV().getId(), a -> a));

                return cvs.stream()
                                .map(cv -> toApplicationSummaryResponse(cv, analysisMap.get(cv.getId())))
                                .collect(Collectors.toList());
        }

        /**
         * Thống kê CV cho HR chatbot: tổng số, đã chấm, pass (>=75), fail.
         * Dùng bởi get_cv_summary tool để trả lời chính xác khi HR hỏi số lượng CV.
         *
         * @param positionId    ID vị trí cần thống kê
         * @param passThreshold ngưỡng điểm pass (mặc định 75)
         */
        public CvStatisticsResponse getCvStatistics(int positionId, int passThreshold, String mode) {
                SourceType sourceType = "INTERNAL".equals(mode)
                                ? SourceType.INTERNAL
                                : SourceType.EXTERNAL;
                long total = candidateCVRepository.countByPositionIdAndSourceType(positionId, sourceType);
                long scored = cvAnalysisRepository.countScoredByPositionIdAndSourceType(positionId, sourceType);
                long passed = cvAnalysisRepository.countPassedByPositionIdAndSourceType(positionId, sourceType,
                                passThreshold);
                return CvStatisticsResponse.builder()
                                .positionId(positionId)
                                .total(total)
                                .scored(scored)
                                .passed(passed)
                                .failed(scored - passed)
                                .build();
        }

        /**
         * Trả về trạng thái ứng tuyển của một candidate — dùng bởi
         * check_application_status tool.
         * Cho phép Candidate chatbot trả lời "Tôi đã apply chưa?" mà không cần điều
         * hướng UI.
         *
         * @param candidateId UUID của ứng viên
         * @param positionId  Optional — nếu có, chỉ trả về application của position đó
         */
        public CandidateApplicationStatusResponse getApplicationStatus(
                        String candidateId, Optional<Integer> positionId) {

                List<CandidateCV> applications = positionId
                                .map(pid -> candidateCVRepository
                                                .findApplicationsByCandidateIdAndPositionId(candidateId, pid))
                                .orElseGet(() -> candidateCVRepository.findApplicationsByCandidateId(candidateId));

                List<CandidateApplicationStatusResponse.ApplicationRecord> records = applications.stream()
                                .map(cv -> {
                                        org.example.recruitmentservice.models.enums.MatchStatus matchStatus = cvAnalysisRepository
                                                        .findByCandidateCV_Id(cv.getId())
                                                        .map(a -> a.getOverallStatus())
                                                        .orElse(null);
                                        Integer score = cvAnalysisRepository.findByCandidateCV_Id(cv.getId())
                                                        .map(a -> a.getTechnicalScore())
                                                        .orElse(null);
                                        String posName = cv.getPosition() != null ? cv.getPosition().getTitle() : null;
                                        String status = matchStatus != null ? matchStatus.name() : "PENDING";
                                        return CandidateApplicationStatusResponse.ApplicationRecord.builder()
                                                        .positionId(cv.getPosition() != null ? cv.getPosition().getId()
                                                                        : null)
                                                        .positionName(posName)
                                                        .score(score)
                                                        .status(status)
                                                        .build();
                                })
                                .collect(Collectors.toList());

                return CandidateApplicationStatusResponse.builder()
                                .candidateId(candidateId)
                                .applications(records)
                                .build();
        }

        /**
         * Cập nhật trạng thái RecruitmentStage của Application CV khi gửi email.
         */
        public void updateRecruitmentStage(Integer appCvId, String emailType, String interviewDate) {
                if (appCvId == null || emailType == null)
                        return;

                candidateCVRepository.findById(appCvId).ifPresent(cv -> {
                        RecruitmentStage newStage = null;
                        switch (emailType.toUpperCase()) {
                                case "INTERVIEW_INVITE":
                                        newStage = RecruitmentStage.INTERVIEW_SCHEDULED;
                                        if (interviewDate != null && !interviewDate.trim().isEmpty()) {
                                                try {
                                                        String dateStr = interviewDate.trim();
                                                        // Normalize space to 'T' for ISO format
                                                        if (dateStr.matches(
                                                                        "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}(:\\d{2})?(.*)?$")) {
                                                                dateStr = dateStr.replaceFirst(" ", "T");
                                                        }

                                                        // If it's just YYYY-MM-DDTHH:MM without seconds, append :00
                                                        if (dateStr.matches("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}$")) {
                                                                dateStr += ":00";
                                                        }

                                                        if (dateStr.contains("T") && (dateStr.endsWith("Z")
                                                                        || dateStr.contains("+")
                                                                        || dateStr.matches(".*-\\d{2}:\\d{2}$"))) {
                                                                cv.setInterviewSchedule(java.time.ZonedDateTime
                                                                                .parse(dateStr)
                                                                                .toLocalDateTime());
                                                        } else if (dateStr.contains("T")) {
                                                                cv.setInterviewSchedule(java.time.LocalDateTime
                                                                                .parse(dateStr));
                                                        } else {
                                                                cv.setInterviewSchedule(java.time.LocalDate
                                                                                .parse(dateStr).atStartOfDay());
                                                        }
                                                } catch (Exception e) {
                                                        System.err.println("Failed to parse interviewDate: "
                                                                        + interviewDate + " - " + e.getMessage());
                                                }
                                        }
                                        break;
                                case "OFFER_LETTER":
                                        newStage = RecruitmentStage.OFFER;
                                        break;
                        }
                        if (newStage != null) {
                                cv.setRecruitmentStage(newStage);
                                candidateCVRepository.save(cv);
                        }
                });
        }

        /**
         * Lưu điểm đánh giá (CVAnalysis) từ LLM cho một Application CV đã tồn tại.
         */
        public void evaluateApplication(EvaluateApplicationRequest request) {
                if (request.getAppCvId() == null)
                        return;

                CandidateCV cv = candidateCVRepository.findById(request.getAppCvId())
                                .orElseThrow(() -> new CustomException(ErrorCode.CV_NOT_FOUND));

                // Nếu CV đã có điểm thì cập nhật, chưa có thì tạo mới
                CVAnalysis analysis = cvAnalysisRepository.findByCandidateCV_Id(cv.getId()).orElse(new CVAnalysis());

                analysis.setCandidateCV(cv);
                analysis.setPositionId(request.getPositionId());
                if (cv.getPosition() != null) {
                        analysis.setPositionName(cv.getPosition().getTitle());
                }

                analysis.setTechnicalScore(request.getTechnicalScore());
                analysis.setExperienceScore(request.getExperienceScore());
                analysis.setOverallStatus(request.getOverallStatus());
                analysis.setAiAssessment(request.getAiAssessment());
                analysis.setLearningPath(request.getLearningPath());
                analysis.setAnalysisMethod("LLM");

                cvAnalysisRepository.save(analysis);
        }

        // -------------------------------------------------------
        // Mapping helpers (private)
        // -------------------------------------------------------

        private PositionDetailsResponse toPositionDetailsResponse(Positions position) {
                return PositionDetailsResponse.builder()
                                .id(position.getId())
                                .title(position.getTitle())
                                .seniority(position.getSeniority())
                                .jdText(position.getJobDescription())
                                .build();
        }

        private ActivePositionResponse toActivePositionResponse(Positions position) {
                String openedAt = position.getOpenedAt() != null
                                ? position.getOpenedAt().toString()
                                : null;
                return ActivePositionResponse.builder()
                                .id(position.getId())
                                .title(position.getTitle())
                                .seniority(position.getSeniority())
                                .skills(position.getSkills())
                                .minimumFitScore(position.getMinimumFitScore())
                                .openedAt(openedAt)
                                .build();
        }

        private ApplicationSummaryResponse toApplicationSummaryResponse(CandidateCV cv, CVAnalysis analysis) {
                ApplicationSummaryResponse.ApplicationSummaryResponseBuilder builder = ApplicationSummaryResponse
                                .builder()
                                .candidateId(cv.getCandidateId())
                                .candidateName(cv.getName())
                                .candidateEmail(cv.getEmail())
                                .appCvId(cv.getId())
                                .masterCvId(cv.getParentCvId())
                                .sourceType(cv.getSourceType() != null ? cv.getSourceType().name() : null);

                if (analysis != null) {
                        builder.score(analysis.getTechnicalScore())
                                        .aiAssessment(analysis.getAiAssessment());
                }

                return builder.build();
        }
}
