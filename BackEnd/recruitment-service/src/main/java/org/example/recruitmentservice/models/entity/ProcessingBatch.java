package org.example.recruitmentservice.models.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.example.recruitmentservice.models.enums.BatchStatus;
import org.example.recruitmentservice.models.enums.BatchType;

import java.time.LocalDateTime;

@Entity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "processing_batch")
@Builder
public class ProcessingBatch {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(nullable = false, unique = true)
    private int id;

    @Column(nullable = false, unique = true)
    private String batchId;

    @Column
    private Integer positionId;

    @Column(nullable = false)
    private Integer total;

    @Column
    private Integer success;

    @Column
    Integer failed;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Builder.Default
    private BatchStatus status = BatchStatus.PROCESSING;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private BatchType type;

    private LocalDateTime createdAt;

    private LocalDateTime completedAt;

    @Transient
    public Integer getProcessed() {
        return success + failed;
    }

    @Transient
    public Double getProgress() {
        return (success + failed) * 100.0 / total;
    }

    @Transient
    public Integer getPending() {
        return total - success - failed;
    }
}
