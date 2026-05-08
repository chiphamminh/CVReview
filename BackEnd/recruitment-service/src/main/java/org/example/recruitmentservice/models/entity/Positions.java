package org.example.recruitmentservice.models.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "positions")
@Builder
@org.hibernate.annotations.DynamicUpdate
public class Positions {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(nullable = false, unique = true)
    private int id;

    @Column(nullable = false)
    private String hrId;

    @OneToMany(mappedBy = "position", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    @Builder.Default
    private List<CandidateCV> candidateCVs = new ArrayList<>();

    private String title;

    @ElementCollection
    @CollectionTable(name = "position_skills", joinColumns = @JoinColumn(name = "position_id"))
    @Column(name = "skill")
    @Builder.Default
    private List<String> skills = new ArrayList<>();

    private String seniority;

    @Column
    private Double minimumFitScore;

    @Lob
    @Column(columnDefinition = "TEXT", nullable = false)
    private String jobDescription;

    @Column
    private String jdPath;

    @Column
    private String driveFileId;

    @Column
    private String driveFileUrl;

    @Column(nullable = false)
    @Builder.Default
    private boolean isActive = true;

    @Column
    private LocalDateTime openedAt;

    @Column
    private LocalDateTime closedAt;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @Column(nullable = false)
    private LocalDateTime updatedAt;

    @Column
    @Enumerated(EnumType.STRING)
    @Builder.Default
    private org.example.recruitmentservice.models.enums.JDStatus status = org.example.recruitmentservice.models.enums.JDStatus.PENDING;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String errorMessage;

    @Column
    private String batchId;
}
