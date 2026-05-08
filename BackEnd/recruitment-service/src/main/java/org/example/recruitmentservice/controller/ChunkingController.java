package org.example.recruitmentservice.controller;

import lombok.RequiredArgsConstructor;
import org.example.recruitmentservice.dto.request.ChunkPayload;
import org.example.recruitmentservice.dto.request.JDChunkPayload;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.entity.Positions;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.example.recruitmentservice.repository.PositionRepository;
import org.example.recruitmentservice.services.chunking.ChunkingService;
import org.example.recruitmentservice.services.chunking.JDChunkingService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.example.recruitmentservice.utils.PositionUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/chunking")
@RequiredArgsConstructor
public class ChunkingController {

    private final CandidateCVRepository cvRepository;
    private final ChunkingService chunkingService;
    private final PositionRepository positionRepository;
    private final JDChunkingService jdChunkingService;

    @PostMapping("")
    public ResponseEntity<?> chunking(@RequestBody Map<String, Integer> request) {
        Integer cvId = request.get("cvId");

        // 1. Lấy CV từ DB
        CandidateCV cv = cvRepository.findById(cvId)
                .orElseThrow(() -> new RuntimeException("CV not found: " + cvId));

        // 2. Check có content không
        if (cv.getCvContent() == null || cv.getCvContent().isBlank()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "CV content is empty"));
        }

        // 3. Chunk
        List<ChunkPayload> chunks = chunkingService.chunk(
                cv,
                cv.getCvContent());

        // 4. Build response
        Map<String, Object> response = new HashMap<>();
        response.put("cvId", cv.getId());
        response.put("candidateId", cv.getCandidateId());
        response.put("candidateName", cv.getName());
        response.put("position",
                cv.getPosition() != null
                        ? PositionUtils.formatPositionTitle(cv.getPosition().getSeniority(),
                                cv.getPosition().getTitle())
                        : null);
        response.put("totalChunks", chunks.size());
        response.put("totalWords", chunks.stream().mapToInt(ChunkPayload::getWords).sum());
        response.put("totalTokens", chunks.stream().mapToInt(ChunkPayload::getTokensEstimate).sum());
        response.put("chunks", chunks);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/{cvId}")
    public ResponseEntity<?> chunkingGet(@PathVariable Integer cvId) {
        return chunking(Map.of("cvId", cvId));
    }

    @PostMapping("/jd")
    public ResponseEntity<?> chunkingJd(@RequestBody Map<String, Integer> request) {
        Integer positionId = request.get("positionId");

        Positions position = positionRepository.findById(positionId)
                .orElseThrow(() -> new RuntimeException("Position not found: " + positionId));

        if (position.getJobDescription() == null || position.getJobDescription().isBlank()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "JD content is empty"));
        }

        List<JDChunkPayload> chunks = jdChunkingService.chunk(
                position.getId(),
                position.getTitle(),
                position.getSeniority(),
                position.getJobDescription());

        Map<String, Object> response = new HashMap<>();
        response.put("positionId", position.getId());
        response.put("positionName", PositionUtils.formatPositionTitle(position.getSeniority(), position.getTitle()));
        response.put("totalChunks", chunks.size());
        response.put("totalWords", chunks.stream().mapToInt(JDChunkPayload::getWords).sum());
        response.put("totalTokens", chunks.stream().mapToInt(JDChunkPayload::getTokensEstimate).sum());
        response.put("chunks", chunks);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/jd/{positionId}")
    public ResponseEntity<?> chunkingJdGet(@PathVariable Integer positionId) {
        return chunkingJd(Map.of("positionId", positionId));
    }
}