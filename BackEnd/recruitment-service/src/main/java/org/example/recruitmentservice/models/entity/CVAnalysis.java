package org.example.recruitmentservice.models.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import org.example.recruitmentservice.models.enums.MatchStatus;

@Entity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "cv_analysis")
public class CVAnalysis {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(nullable = false, unique = true)
    private int id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "cv_id", nullable = false)
    private CandidateCV candidateCV;

    @Column
    private Integer positionId;

    @Column
    private String positionName;

    private Integer technicalScore;

    private Integer experienceScore;

    @Enumerated(EnumType.STRING)
    private MatchStatus overallStatus;

    @Column(columnDefinition = "TEXT")
    private String aiAssessment;

    @Column(columnDefinition = "TEXT")
    private String learningPath;

    @Column
    private String analysisMethod = "LLM";
}