package org.example.recruitmentservice.controller;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.dto.response.PageResponse;
import org.example.recruitmentservice.dto.request.PositionsRequest;
import org.example.recruitmentservice.dto.response.PositionsResponse;
import org.example.recruitmentservice.services.PositionService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/positions")
@RequiredArgsConstructor
public class PositionController {
    private final PositionService positionService;

    @PreAuthorize("hasRole('HR')")
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<PositionsResponse>> createPosition(
            @ModelAttribute PositionsRequest positionsRequest,
            HttpServletRequest request) {
        return ResponseEntity.ok(positionService.createPosition(positionsRequest, request));
    }

    @PreAuthorize("hasAnyRole('HR', 'CANDIDATE')")
    @GetMapping
    public ResponseEntity<ApiResponse<PageResponse<PositionsResponse>>> filterPositions(
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Boolean isActive,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        return ResponseEntity.ok(positionService.filterPositions(keyword, isActive, page, size));
    }

    @PreAuthorize("hasRole('HR')")
    @PatchMapping("/{positionId}/minimum-fit-score")
    public ResponseEntity<ApiResponse<Object>> updateMinimumFitScore(
            @PathVariable int positionId,
            @RequestParam double score) {
        positionService.updateMinimumFitScore(positionId, score);
        return ResponseEntity.ok(new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Updated successfully"));
    }

    @PreAuthorize("hasRole('HR')")
    @PatchMapping("/{positionId}/toggle-active")
    public ResponseEntity<ApiResponse<Object>> toggleActiveStatus(@PathVariable int positionId) {
        positionService.toggleActiveStatus(positionId);
        return ResponseEntity.ok(new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "Updated successfully"));
    }

    @GetMapping("/jd/{positionId}/text")
    public ResponseEntity<ApiResponse<PositionsResponse>> getJdText(
            @PathVariable int positionId) {
        return ResponseEntity.ok(positionService.getJdText(positionId));
    }

    @PreAuthorize("hasRole('HR')")
    @PutMapping("/{positionId}")
    public ResponseEntity<ApiResponse<Object>> updatePosition(
            @PathVariable int positionId,
            @ModelAttribute PositionsRequest positionsRequest) {
        positionService.updatePosition(positionId, positionsRequest);
        return ResponseEntity.ok(new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Updated successfully"));
    }

    @PreAuthorize("hasRole('HR')")
    @DeleteMapping("")
    public ResponseEntity<ApiResponse<Object>> deletePosition(
            @RequestBody List<Integer> positionIds) {
        positionService.deletePositions(positionIds);
        return ResponseEntity.ok(new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Deleted successfully"));
    }
}
