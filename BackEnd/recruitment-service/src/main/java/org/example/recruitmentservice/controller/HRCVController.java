package org.example.recruitmentservice.controller;

import lombok.RequiredArgsConstructor;
import org.example.commonlibrary.dto.response.ApiResponse;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.recruitmentservice.dto.response.AdminCvSummaryDto;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.enums.SourceType;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/hr/cvs")
@RequiredArgsConstructor
public class HRCVController {

    private final CandidateCVRepository candidateCVRepository;

    @GetMapping
    public ApiResponse<Page<AdminCvSummaryDto>> getCvList(
            @RequestParam(required = false) SourceType sourceType,
            @RequestParam(required = false) CVStatus cvStatus,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "updatedAt") String sortBy,
            @RequestParam(defaultValue = "DESC") String direction) {
        try {
            Sort.Direction sortDirection = Sort.Direction.fromString(direction);
            Pageable pageable = PageRequest.of(page, size, Sort.by(sortDirection, sortBy));

            Page<AdminCvSummaryDto> cvPage = candidateCVRepository.findAdminCvList(sourceType, cvStatus, pageable);

            return new ApiResponse<>(ErrorCode.SUCCESS.getCode(), "CV list retrieved successfully", cvPage);
        } catch (Exception e) {
            e.printStackTrace();
            return new ApiResponse<>(ErrorCode.INTERNAL_SERVER_ERROR.getCode(), "Failed to get CV list", null);
        }
    }
}
