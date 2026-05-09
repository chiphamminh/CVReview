package org.example.recruitmentservice.services;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.example.commonlibrary.exception.CustomException;
import org.example.commonlibrary.dto.response.ErrorCode;
import org.example.recruitmentservice.dto.request.InterviewNotificationRequest;
import org.example.recruitmentservice.models.enums.EmailType;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import org.thymeleaf.TemplateEngine;
import org.thymeleaf.context.Context;

/**
 * Gửi email SMTP qua Spring Mail + Thymeleaf template.
 * Hỗ trợ 3 loại: INTERVIEW_INVITE, OFFER_LETTER, REJECTION.
 * Mỗi loại có template HTML riêng trong resources/templates/email/.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationService {

    private final JavaMailSender mailSender;
    private final TemplateEngine templateEngine;

    @Value("${chatbot.from-email}")
    private String fromEmail;

    public void sendInterviewNotification(InterviewNotificationRequest request) {
        EmailType emailType = parseEmailType(request.getEmailType());

        String subject = buildSubject(emailType, request.getPositionName());
        String templateName = resolveTemplateName(emailType);
        String htmlContent = buildHtmlContent(templateName, request);

        sendHtmlEmail(request.getCandidateEmail(), subject, htmlContent);
        log.info("Email {} sent to {} for position {}",
                emailType, request.getCandidateEmail(), request.getPositionName());
    }

    private EmailType parseEmailType(String emailTypeStr) {
        try {
            return EmailType.valueOf(emailTypeStr.toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new CustomException(ErrorCode.MISSING_REQUIRED_FIELD);
        }
    }

    private String buildSubject(EmailType emailType, String positionName) {
        return switch (emailType) {
            case INTERVIEW_INVITE -> "[CV Review] Thư Mời Phỏng Vấn — " + positionName;
            case OFFER_LETTER     -> "[CV Review] Chúc Mừng Trúng Tuyển — " + positionName;
        };
    }

    private String resolveTemplateName(EmailType emailType) {
        return switch (emailType) {
            case INTERVIEW_INVITE -> "email/interview-invite";
            case OFFER_LETTER     -> "email/offer-letter";
        };
    }

    private String buildHtmlContent(String templateName, InterviewNotificationRequest request) {
        Context ctx = new Context();
        ctx.setVariable("candidateName", request.getCandidateName());
        ctx.setVariable("positionName", request.getPositionName());
        ctx.setVariable("interviewDate", request.getInterviewDate());
        ctx.setVariable("customMessage", request.getCustomMessage());
        ctx.setVariable("benefit", request.getBenefit());
        ctx.setVariable("salary", request.getSalary());
        ctx.setVariable("startDate", request.getStartDate());
        ctx.setVariable("offerExpirationDate", request.getOfferExpirationDate());
        ctx.setVariable("additionalNote", request.getAdditionalNote());
        return templateEngine.process(templateName, ctx);
    }

    private void sendHtmlEmail(String to, String subject, String htmlContent) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
            helper.setFrom(fromEmail);
            helper.setTo(to);
            helper.setSubject(subject);
            helper.setText(htmlContent, true);
            mailSender.send(message);
        } catch (MessagingException e) {
            log.error("Failed to send email to {}: {}", to, e.getMessage());
            throw new CustomException(ErrorCode.EMAIL_SEND_FAILED);
        }
    }
}
