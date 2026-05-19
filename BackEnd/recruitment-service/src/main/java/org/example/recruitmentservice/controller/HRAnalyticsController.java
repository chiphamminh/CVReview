package org.example.recruitmentservice.controller;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.recruitmentservice.dto.response.*;
import org.example.recruitmentservice.scheduler.GarbageCollectionJob;
import org.example.recruitmentservice.services.HRAnalyticsService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/hr/analytics")
@RequiredArgsConstructor
public class HRAnalyticsController {

    private final HRAnalyticsService hrAnalyticsService;
    private final GarbageCollectionJob garbageCollectionJob;

    @GetMapping("/active-positions")
    public ApiResponse<List<ActivePositionResponse>> getActivePositions() {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Active positions retrieved",
                    hrAnalyticsService.getActivePositions());
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get active positions", null);
        }
    }

    @GetMapping("/cv-traffic")
    public ApiResponse<CvTrafficResponse> getCvTraffic(
            @RequestParam(value = "days", defaultValue = "30") int days,
            @RequestParam(value = "positionId", required = false) Integer positionId) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "CV traffic retrieved successfully",
                    hrAnalyticsService.getCvTraffic(days, positionId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get CV traffic", null);
        }
    }

    @GetMapping("/processing-time")
    public ApiResponse<ProcessingTimeResponse> getAverageProcessingTime(
            @RequestParam(value = "days", defaultValue = "30") int days) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Processing time retrieved successfully",
                    hrAnalyticsService.getAverageProcessingTime(days));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get processing time", null);
        }
    }

    @GetMapping("/overview")
    public ApiResponse<OverviewResponse> getOverview(
            @RequestParam(value = "days", defaultValue = "30") int days,
            @RequestParam(value = "positionId", required = false) Integer positionId) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Overview retrieved successfully",
                    hrAnalyticsService.getOverview(days, positionId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get overview", null);
        }
    }

    @GetMapping("/score-distribution")
    public ApiResponse<ScoreDistributionResponse> getScoreDistribution(
            @RequestParam(value = "positionId", required = false) Integer positionId) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Score distribution retrieved successfully",
                    hrAnalyticsService.getScoreDistribution(positionId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get score distribution", null);
        }
    }

    @GetMapping("/stage-pipeline")
    public ApiResponse<StagePipelineResponse> getStagePipeline(
            @RequestParam(value = "positionId", required = false) Integer positionId) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Stage pipeline retrieved successfully",
                    hrAnalyticsService.getStagePipeline(positionId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get stage pipeline", null);
        }
    }

    @GetMapping("/source-breakdown")
    public ApiResponse<SourceBreakdownResponse> getSourceBreakdown(
            @RequestParam(value = "days", defaultValue = "30") int days,
            @RequestParam(value = "positionId", required = false) Integer positionId) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Source breakdown retrieved successfully",
                    hrAnalyticsService.getSourceBreakdown(days, positionId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get source breakdown", null);
        }
    }

    @GetMapping("/score-trend")
    public ApiResponse<ScoreTrendResponse> getScoreTrend(
            @RequestParam(value = "days", defaultValue = "30") int days,
            @RequestParam(value = "positionId", required = false) Integer positionId) {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Score trend retrieved successfully",
                    hrAnalyticsService.getScoreTrend(days, positionId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get score trend", null);
        }
    }

    @GetMapping("/positions-health")
    public ApiResponse<PositionsHealthResponse> getPositionsHealth() {
        try {
            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Positions health retrieved successfully",
                    hrAnalyticsService.getPositionsHealth());
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get positions health", null);
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
