package org.example.recruitmentservice.models.enums;

/**
 * Mode hoạt động của HR chatbot.
 * INTERNAL: làm việc với CVs do HR upload (sourceType=HR).
 * EXTERNAL: làm việc với CVs do Candidate nộp vào (sourceType=CANDIDATE).
 */
public enum ChatMode {
    INTERNAL,
    EXTERNAL
}
