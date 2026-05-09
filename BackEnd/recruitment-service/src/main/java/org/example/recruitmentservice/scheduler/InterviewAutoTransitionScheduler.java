package org.example.recruitmentservice.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.recruitmentservice.models.entity.CandidateCV;
import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.repository.CandidateCVRepository;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
public class InterviewAutoTransitionScheduler {

    private final CandidateCVRepository candidateCVRepository;

    @Scheduled(fixedDelay = 900_000)
    @Transactional
    public void autoTransitionInterviewedCandidates() {
        List<CandidateCV> pastDue = candidateCVRepository.findInterviewsPastDue(
                RecruitmentStage.INTERVIEW_SCHEDULED, LocalDateTime.now());

        if (pastDue.isEmpty()) {
            return;
        }

        pastDue.forEach(cv -> cv.setRecruitmentStage(RecruitmentStage.INTERVIEWED));
        candidateCVRepository.saveAll(pastDue);

        log.info("Auto-transitioned {} candidate(s) from INTERVIEW_SCHEDULED to INTERVIEWED",
                pastDue.size());
    }
}
