package org.example.recruitmentservice.services;

import lombok.RequiredArgsConstructor;
import org.example.recruitmentservice.dto.response.*;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.models.enums.SourceType;
import org.example.recruitmentservice.repository.CVAnalysisRepository;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.example.recruitmentservice.repository.ProcessingBatchRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class HRAnalyticsService {

    private final CandidateCVRepository candidateCVRepository;
    private final ProcessingBatchRepository processingBatchRepository;
    private final CVAnalysisRepository cvAnalysisRepository;
    private final PositionRepository positionRepository;

    public List<ActivePositionResponse> getActivePositions() {
        return positionRepository.findAllActive().stream()
                .map(p -> ActivePositionResponse.builder()
                        .id(p.getId())
                        .title(p.getTitle())
                        .seniority(p.getSeniority())
                        .minimumFitScore(p.getMinimumFitScore())
                        .build())
                .collect(Collectors.toList());
    }

    public CvTrafficResponse getCvTraffic(int days, Integer positionId) {
        LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

        long totalCv, successCv, failedCv;
        if (positionId != null) {
            totalCv    = candidateCVRepository.countTotalByPositionAfterDate(positionId, dateSince);
            successCv  = candidateCVRepository.countByPositionAndStatusAfterDate(positionId, CVStatus.EMBEDDED, dateSince);
            failedCv   = candidateCVRepository.countByPositionAndStatusAfterDate(positionId, CVStatus.FAILED, dateSince);
        } else {
            totalCv    = candidateCVRepository.countTotalCVsAfterDate(dateSince);
            successCv  = candidateCVRepository.countByCvStatusAndDateAfter(CVStatus.EMBEDDED, dateSince);
            failedCv   = candidateCVRepository.countByCvStatusAndDateAfter(CVStatus.FAILED, dateSince);
        }

        long processingCv = Math.max(0, totalCv - successCv - failedCv);
        return CvTrafficResponse.builder()
                .totalCv(totalCv)
                .successCv(successCv)
                .failedCv(failedCv)
                .processingCv(processingCv)
                .days(days)
                .build();
    }

    public ProcessingTimeResponse getAverageProcessingTime(int days) {
        LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

        Double t1 = processingBatchRepository.getAverageTimeFor1To10CVs(dateSince);
        Double t2 = processingBatchRepository.getAverageTimeFor11To20CVs(dateSince);
        Double t3 = processingBatchRepository.getAverageTimeFor21To30CVs(dateSince);
        Double t4 = processingBatchRepository.getAverageTimeForMoreThan30CVs(dateSince);

        List<ProcessingTimeResponse.BucketTime> buckets = new ArrayList<>();
        if (t1 != null) buckets.add(new ProcessingTimeResponse.BucketTime("1-10 CVs", t1));
        if (t2 != null) buckets.add(new ProcessingTimeResponse.BucketTime("11-20 CVs", t2));
        if (t3 != null) buckets.add(new ProcessingTimeResponse.BucketTime("21-30 CVs", t3));
        if (t4 != null) buckets.add(new ProcessingTimeResponse.BucketTime("> 30 CVs", t4));

        return ProcessingTimeResponse.builder().days(days).buckets(buckets).build();
    }

    public OverviewResponse getOverview(int days, Integer positionId) {
        LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

        long totalCvsScored, totalCvsPassed;
        Double avgRaw;

        if (positionId != null) {
            Positions position = positionRepository.findById(positionId.intValue());
            if (position == null) throw new IllegalArgumentException("Position not found: " + positionId);
            int minSum = toMinSum(position.getMinimumFitScore());
            totalCvsScored = cvAnalysisRepository.countScoredByPositionAfterDate(positionId, dateSince);
            totalCvsPassed = cvAnalysisRepository.countPassedByPositionAfterDate(positionId, minSum, dateSince);
            avgRaw         = cvAnalysisRepository.averageScoreByPositionAfterDate(positionId, dateSince);
        } else {
            totalCvsScored = cvAnalysisRepository.countScoredAfterDate(dateSince);
            totalCvsPassed = cvAnalysisRepository.countPassedAfterDate(dateSince);
            avgRaw         = cvAnalysisRepository.averageCompositeScoreAfterDate(dateSince);
        }

        double avgMatchingScore = round1(avgRaw);
        double successMatchRate = totalCvsScored > 0
                ? Math.round((totalCvsPassed * 100.0 / totalCvsScored) * 10.0) / 10.0 : 0.0;

        return OverviewResponse.builder()
                .totalCvsScored(totalCvsScored)
                .totalCvsPassed(totalCvsPassed)
                .avgMatchingScore(avgMatchingScore)
                .successMatchRate(successMatchRate)
                .days(days)
                .build();
    }

    public ScoreDistributionResponse getScoreDistribution(Integer positionId) {
        List<ScoreDistributionResponse.ScoreBucket> buckets;
        if (positionId != null) {
            buckets = List.of(
                    bucket("0-20",   "Very Weak",  cvAnalysisRepository.countInRangeByPosition(positionId,   0,  42)),
                    bucket("21-40",  "Weak",        cvAnalysisRepository.countInRangeByPosition(positionId,  42,  82)),
                    bucket("41-60",  "Average",     cvAnalysisRepository.countInRangeByPosition(positionId,  82, 122)),
                    bucket("61-80",  "Good",        cvAnalysisRepository.countInRangeByPosition(positionId, 122, 162)),
                    bucket("81-100", "Excellent",   cvAnalysisRepository.countInRangeByPosition(positionId, 162, 202)));
        } else {
            buckets = List.of(
                    bucket("0-20",   "Very Weak",  cvAnalysisRepository.countInCompositeRange(  0,  42)),
                    bucket("21-40",  "Weak",        cvAnalysisRepository.countInCompositeRange( 42,  82)),
                    bucket("41-60",  "Average",     cvAnalysisRepository.countInCompositeRange( 82, 122)),
                    bucket("61-80",  "Good",        cvAnalysisRepository.countInCompositeRange(122, 162)),
                    bucket("81-100", "Excellent",   cvAnalysisRepository.countInCompositeRange(162, 202)));
        }
        return ScoreDistributionResponse.builder().buckets(buckets).build();
    }

    public StagePipelineResponse getStagePipeline(Integer positionId) {
        long applied, interviewScheduled, interviewed, offer, accepted, rejected;

        if (positionId != null) {
            applied             = candidateCVRepository.countByStageAndPositionId(RecruitmentStage.APPLIED, positionId);
            interviewScheduled  = candidateCVRepository.countByStageAndPositionId(RecruitmentStage.INTERVIEW_SCHEDULED, positionId);
            interviewed         = candidateCVRepository.countByStageAndPositionId(RecruitmentStage.INTERVIEWED, positionId);
            offer               = candidateCVRepository.countByStageAndPositionId(RecruitmentStage.OFFER, positionId);
            accepted            = candidateCVRepository.countByStageAndPositionId(RecruitmentStage.ACCEPTED, positionId);
            rejected            = candidateCVRepository.countByStageAndPositionId(RecruitmentStage.REJECTED, positionId);
        } else {
            applied             = candidateCVRepository.countByStageAllPositions(RecruitmentStage.APPLIED);
            interviewScheduled  = candidateCVRepository.countByStageAllPositions(RecruitmentStage.INTERVIEW_SCHEDULED);
            interviewed         = candidateCVRepository.countByStageAllPositions(RecruitmentStage.INTERVIEWED);
            offer               = candidateCVRepository.countByStageAllPositions(RecruitmentStage.OFFER);
            accepted            = candidateCVRepository.countByStageAllPositions(RecruitmentStage.ACCEPTED);
            rejected            = candidateCVRepository.countByStageAllPositions(RecruitmentStage.REJECTED);
        }

        return StagePipelineResponse.builder()
                .applied(applied)
                .interviewScheduled(interviewScheduled)
                .interviewed(interviewed)
                .offer(offer)
                .accepted(accepted)
                .rejected(rejected)
                .positionId(positionId)
                .build();
    }

    public SourceBreakdownResponse getSourceBreakdown(int days, Integer positionId) {
        long internalCount, internalScored, internalPassed;
        long externalCount, externalScored, externalPassed;
        double internalAvgScore, externalAvgScore;

        if (positionId != null) {
            Positions position = positionRepository.findById(positionId.intValue());
            int minSum = (position != null) ? toMinSum(position.getMinimumFitScore()) : 140;

            internalCount  = candidateCVRepository.countActiveByPositionAndSourceType(positionId, SourceType.INTERNAL);
            externalCount  = candidateCVRepository.countActiveByPositionAndSourceType(positionId, SourceType.EXTERNAL);
            internalScored = cvAnalysisRepository.countScoredByPositionAndSource(positionId, SourceType.INTERNAL);
            externalScored = cvAnalysisRepository.countScoredByPositionAndSource(positionId, SourceType.EXTERNAL);
            internalPassed = cvAnalysisRepository.countPassedByPositionAndSource(positionId, SourceType.INTERNAL, minSum);
            externalPassed = cvAnalysisRepository.countPassedByPositionAndSource(positionId, SourceType.EXTERNAL, minSum);
            internalAvgScore = round1(cvAnalysisRepository.avgScoreByPositionAndSource(positionId, SourceType.INTERNAL));
            externalAvgScore = round1(cvAnalysisRepository.avgScoreByPositionAndSource(positionId, SourceType.EXTERNAL));
        } else {
            LocalDateTime dateSince = LocalDateTime.now().minusDays(days);
            internalCount  = candidateCVRepository.countBySourceTypeAfterDate(SourceType.INTERNAL, dateSince);
            externalCount  = candidateCVRepository.countBySourceTypeAfterDate(SourceType.EXTERNAL, dateSince);
            internalScored = cvAnalysisRepository.countScoredBySourceAfterDate(SourceType.INTERNAL, dateSince);
            externalScored = cvAnalysisRepository.countScoredBySourceAfterDate(SourceType.EXTERNAL, dateSince);
            internalPassed = cvAnalysisRepository.countPassedBySourceAfterDate(SourceType.INTERNAL, dateSince);
            externalPassed = cvAnalysisRepository.countPassedBySourceAfterDate(SourceType.EXTERNAL, dateSince);
            internalAvgScore = round1(cvAnalysisRepository.avgScoreBySourceAfterDate(SourceType.INTERNAL, dateSince));
            externalAvgScore = round1(cvAnalysisRepository.avgScoreBySourceAfterDate(SourceType.EXTERNAL, dateSince));
        }

        return SourceBreakdownResponse.builder()
                .internalCount(internalCount)
                .internalScored(internalScored)
                .internalPassed(internalPassed)
                .internalAvgScore(internalAvgScore)
                .externalCount(externalCount)
                .externalScored(externalScored)
                .externalPassed(externalPassed)
                .externalAvgScore(externalAvgScore)
                .days(days)
                .positionId(positionId)
                .build();
    }

    public ScoreTrendResponse getScoreTrend(int days, Integer positionId) {
        LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

        List<ScoreTrendProjection> projections = positionId != null
                ? cvAnalysisRepository.getScoreTrendByPosition(positionId, dateSince)
                : cvAnalysisRepository.getScoreTrendAllPositions(dateSince);

        List<ScoreTrendResponse.WeekPoint> points = projections.stream()
                .map(p -> new ScoreTrendResponse.WeekPoint(
                        String.format("%d-W%02d", p.getYr(), p.getWk()),
                        p.getAvgScore() != null ? Math.round(p.getAvgScore() * 10.0) / 10.0 : 0.0,
                        p.getCvCount() != null ? p.getCvCount() : 0L))
                .collect(Collectors.toList());

        return ScoreTrendResponse.builder()
                .points(points)
                .positionId(positionId)
                .days(days)
                .build();
    }

    public PositionsHealthResponse getPositionsHealth() {
        List<PositionsHealthResponse.PositionHealth> healthList = positionRepository.findAllActive().stream()
                .map(p -> {
                    int posId = p.getId();
                    double threshold = p.getMinimumFitScore() != null ? p.getMinimumFitScore() : 70.0;
                    int minSum = (int) Math.round(threshold * 2);

                    long poolSize      = candidateCVRepository.countActiveByPositionId(posId);
                    Double avgRaw      = cvAnalysisRepository.averageScoreByPosition(posId);
                    long qualifiedCount = cvAnalysisRepository.countQualifiedByPosition(posId, minSum);
                    double avgScore    = round1(avgRaw);

                    String healthStatus = qualifiedCount >= 10 ? "HEALTHY"
                            : qualifiedCount >= 4 ? "MEDIUM" : "LOW";

                    return PositionsHealthResponse.PositionHealth.builder()
                            .id(posId)
                            .title(p.getTitle())
                            .seniority(p.getSeniority())
                            .minimumFitScore(threshold)
                            .poolSize(poolSize)
                            .avgScore(avgScore)
                            .qualifiedCount(qualifiedCount)
                            .healthStatus(healthStatus)
                            .build();
                })
                .collect(Collectors.toList());

        return PositionsHealthResponse.builder().positions(healthList).build();
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static int toMinSum(Double minimumFitScore) {
        return minimumFitScore != null ? (int) Math.round(minimumFitScore * 2) : 140;
    }

    private static double round1(Double value) {
        return value != null ? Math.round(value * 10.0) / 10.0 : 0.0;
    }

    private static ScoreDistributionResponse.ScoreBucket bucket(String range, String label, long count) {
        return new ScoreDistributionResponse.ScoreBucket(range, label, count);
    }
}
