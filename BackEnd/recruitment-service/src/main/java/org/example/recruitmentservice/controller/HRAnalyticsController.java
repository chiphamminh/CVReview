package org.example.recruitmentservice.controller;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.recruitmentservice.dto.response.CvTrafficResponse;
import org.example.recruitmentservice.dto.response.OverviewResponse;
import org.example.recruitmentservice.dto.response.ProcessingTimeResponse;
import org.example.recruitmentservice.dto.response.ScoreDistributionResponse;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.repository.CVAnalysisRepository;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.ProcessingBatchRepository;
import org.example.recruitmentservice.scheduler.GarbageCollectionJob;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/hr/analytics")
@RequiredArgsConstructor
public class HRAnalyticsController {

    private final CandidateCVRepository candidateCVRepository;
    private final ProcessingBatchRepository processingBatchRepository;
    private final CVAnalysisRepository cvAnalysisRepository;
    private final GarbageCollectionJob garbageCollectionJob;

    @GetMapping("/cv-traffic")
    public ApiResponse<CvTrafficResponse> getCvTraffic(
            @RequestParam(value = "days", defaultValue = "30") int days) {
        try {
            LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

            long totalCv = candidateCVRepository.countTotalCVsAfterDate(dateSince);
            long successCv = candidateCVRepository.countByCvStatusAndDateAfter(CVStatus.EMBEDDED, dateSince);
            long failedCv = candidateCVRepository.countByCvStatusAndDateAfter(CVStatus.FAILED, dateSince);

            long processingCv = totalCv - successCv - failedCv;
            if (processingCv < 0)
                processingCv = 0;

            CvTrafficResponse response = CvTrafficResponse.builder()
                    .totalCv(totalCv)
                    .successCv(successCv)
                    .failedCv(failedCv)
                    .processingCv(processingCv)
                    .days(days)
                    .build();

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "CV traffic retrieved successfully", response);
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get CV traffic", null);
        }
    }

    @GetMapping("/processing-time")
    public ApiResponse<ProcessingTimeResponse> getAverageProcessingTime(
            @RequestParam(value = "days", defaultValue = "30") int days) {
        try {
            LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

            Double t1 = processingBatchRepository.getAverageTimeFor1To10CVs(dateSince);
            Double t2 = processingBatchRepository.getAverageTimeFor11To20CVs(dateSince);
            Double t3 = processingBatchRepository.getAverageTimeFor21To30CVs(dateSince);
            Double t4 = processingBatchRepository.getAverageTimeForMoreThan30CVs(dateSince);

            List<ProcessingTimeResponse.BucketTime> buckets = new ArrayList<>();
            if (t1 != null)
                buckets.add(new ProcessingTimeResponse.BucketTime("1-10 CVs", t1));
            if (t2 != null)
                buckets.add(new ProcessingTimeResponse.BucketTime("11-20 CVs", t2));
            if (t3 != null)
                buckets.add(new ProcessingTimeResponse.BucketTime("21-30 CVs", t3));
            if (t4 != null)
                buckets.add(new ProcessingTimeResponse.BucketTime("> 30 CVs", t4));

            ProcessingTimeResponse response = ProcessingTimeResponse.builder()
                    .days(days)
                    .buckets(buckets)
                    .build();

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Processing time retrieved successfully", response);
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get processing time", null);
        }
    }

    @GetMapping("/overview")
    public ApiResponse<OverviewResponse> getOverview(
            @RequestParam(value = "days", defaultValue = "30") int days) {
        try {
            LocalDateTime dateSince = LocalDateTime.now().minusDays(days);

            long totalCvsScored = cvAnalysisRepository.countScoredAfterDate(dateSince);
            long totalCvsPassed = cvAnalysisRepository.countPassedAfterDate(dateSince);
            Double avgRaw = cvAnalysisRepository.averageCompositeScoreAfterDate(dateSince);

            double avgMatchingScore = avgRaw != null ? Math.round(avgRaw * 10.0) / 10.0 : 0.0;
            double successMatchRate = totalCvsScored > 0
                    ? Math.round((totalCvsPassed * 100.0 / totalCvsScored) * 10.0) / 10.0
                    : 0.0;

            OverviewResponse response = OverviewResponse.builder()
                    .totalCvsScored(totalCvsScored)
                    .totalCvsPassed(totalCvsPassed)
                    .avgMatchingScore(avgMatchingScore)
                    .successMatchRate(successMatchRate)
                    .days(days)
                    .build();

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Overview retrieved successfully", response);
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get overview", null);
        }
    }

    @GetMapping("/score-distribution")
    public ApiResponse<ScoreDistributionResponse> getScoreDistribution() {
        try {
            // Buckets dùng sum = technicalScore + experienceScore (tránh floating-point trong JPQL).
            // composite = sum / 2.0 → bucket 0-20 = sum [0, 42), 21-40 = sum [42, 82), v.v.
            List<ScoreDistributionResponse.ScoreBucket> buckets = List.of(
                    new ScoreDistributionResponse.ScoreBucket("0-20", "Rất yếu",
                            cvAnalysisRepository.countInCompositeRange(0, 42)),
                    new ScoreDistributionResponse.ScoreBucket("21-40", "Yếu",
                            cvAnalysisRepository.countInCompositeRange(42, 82)),
                    new ScoreDistributionResponse.ScoreBucket("41-60", "Trung bình",
                            cvAnalysisRepository.countInCompositeRange(82, 122)),
                    new ScoreDistributionResponse.ScoreBucket("61-80", "Khá",
                            cvAnalysisRepository.countInCompositeRange(122, 162)),
                    new ScoreDistributionResponse.ScoreBucket("81-100", "Xuất sắc",
                            cvAnalysisRepository.countInCompositeRange(162, 202))
            );

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Score distribution retrieved successfully",
                    ScoreDistributionResponse.builder().buckets(buckets).build());
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get score distribution", null);
        }
    }

    @PostMapping("/trigger-gc")
    public ApiResponse<String> triggerGarbageCollection() {
        try {
            garbageCollectionJob.purgeFailedCVFiles();
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Garbage collection triggered manually", "Success");
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to trigger GC", null);
        }
    }
}
