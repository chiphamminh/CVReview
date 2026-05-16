package org.example.recruitmentservice.models.entity;

import jakarta.persistence.*;
import lombok.*;
import org.example.recruitmentservice.models.enums.CVStatus;
import org.example.recruitmentservice.models.enums.RecruitmentStage;
import org.example.recruitmentservice.models.enums.SourceType;

import java.time.LocalDateTime;

@Entity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "candidate_cv")
@org.hibernate.annotations.DynamicUpdate
public class CandidateCV {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(nullable = false, unique = true)
    private int id;

    @Column
    private String candidateId;

    @Column
    private String hrId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "position_id")
    private Positions position;

    /**
     * FK trỏ về Master CV (row có positionId=NULL).
     * NULL đối với Master CV và HR-sourced CVs.
     * Được điền khi Candidate nộp đơn ứng tuyển qua chatbot (finalize_application).
     */
    @Column(name = "parent_cv_id")
    private Integer parentCvId;

    @Column
    @Enumerated(EnumType.STRING)
    private SourceType sourceType;

    @OneToOne(mappedBy = "candidateCV", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private CVAnalysis analysis;

    @Column
    private String email;

    @Column
    private String name;

    @Column
    private String driveFileId;

    @Column
    private String driveFileUrl;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String cvContent;

    @Column(nullable = false)
    private LocalDateTime updatedAt;

    @Column
    @Enumerated(EnumType.STRING)
    private CVStatus cvStatus;

    @Column(length = 50)
    @Enumerated(EnumType.STRING)
    private RecruitmentStage recruitmentStage = RecruitmentStage.APPLIED;

    @Column
    private LocalDateTime createdAt;

    @Column
    private String batchId;

    @Column
    private String fileName;

    @Column
    private LocalDateTime appliedDate;

    @Column
    private LocalDateTime interviewSchedule;

    @Column
    private LocalDateTime failedAt;

    @Column
    private Integer retryCount;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String errorMessage;

    /** Timestamp when the Drive file was physically deleted by the GC job. Null means file still exists on Drive. */
    @Column
    private LocalDateTime deletedAt;
}
