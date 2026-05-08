package org.example.recruitmentservice.controller;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.commonlibrary.dto.response.PageResponse;
import org.example.recruitmentservice.dto.response.CandidateCVResponse;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.services.CandidateCVService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/cv")
@RequiredArgsConstructor
public class CandidateCVController {
    private final CandidateCVService candidateCVService;

    @PreAuthorize("hasRole('HR')")
    @GetMapping("/{cvId}")
    public ApiResponse<CandidateCVResponse> getCVDetail(@PathVariable int cvId) {
        return candidateCVService.getCVDetail(cvId);
    }

    @PreAuthorize("hasRole('HR')")
    @GetMapping("/position/{positionId}")
    public ApiResponse<PageResponse<CandidateCVResponse>> getAllCVsByPositionId(
            @PathVariable int positionId,
            @RequestParam(required = false) List<CVStatus> statuses,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        return candidateCVService.getAllCVsByPositionId(positionId, statuses, page, size);
    }

    @PreAuthorize("hasRole('CANDIDATE')")
    @PutMapping("/{cvId}")
    public ResponseEntity<ApiResponse<Object>> updateCandidateCV(
            @PathVariable int cvId,
            @RequestParam(value = "name", required = false) String name,
            @RequestParam(value = "email", required = false) String email) {
        candidateCVService.updateCV(cvId, name, email);
        return ResponseEntity.ok(new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Updated Candidate CV successfully"));
    }

    @PreAuthorize("hasAnyRole('HR', 'CANDIDATE')")
    @DeleteMapping("")
    public ResponseEntity<ApiResponse<Object>> deleteCandidateCVs(
            @RequestBody List<Integer> cvIds) {
        candidateCVService.deleteCandidateCVs(cvIds);
        return ResponseEntity.ok(new ApiResponse<>(
                ErrorCode.SUCCESS.getCode(),
                "Deleted Candidate CV successfully"));
    }
}
