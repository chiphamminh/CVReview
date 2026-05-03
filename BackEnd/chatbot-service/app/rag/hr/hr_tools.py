"""
Function-calling tools for the HR chatbot.
Registered with Gemini via bind_tools() and executed inside llm_hr_reasoning_node.

Design rules:
- position_id is always auto-injected from HRChatState — never ask LLM to fabricate it.
- candidate_id is NEVER exposed in tool outputs — use name + email only.
- send_interview_email is NEVER called directly — hr_graph handles confirmation first.
"""

from langchain_core.tools import tool
from typing import Optional
from app.services.recruitment_api import recruitment_api


@tool
async def get_candidate_details(candidate_id: str, position_id: int) -> str:
    """
    Lấy thông tin chi tiết score/feedback của một ứng viên cụ thể từ database.
    Gọi khi HR hỏi về điểm số, nhận xét, hoặc kỹ năng cụ thể của một ứng viên.

    Args:
        candidate_id: UUID của ứng viên (tự động inject từ cv_id_to_meta trong graph)
        position_id:  ID của vị trí ứng tuyển (tự động inject từ session)
    """
    try:
        details = await recruitment_api.get_candidate_details(
            candidate_id=candidate_id,
            position_id=position_id
        )
        if not details:
            return "Không tìm thấy thông tin chi tiết cho ứng viên này."

        score     = details.get("score", "N/A")
        feedback  = details.get("feedback", "Không có nhận xét.")
        skill_match = details.get("skillMatch", "N/A")
        skill_miss  = details.get("skillMiss", "N/A")

        return (
            f"**Điểm AI:** {score}/100\n"
            f"**Nhận xét:** {feedback}\n"
            f"**Kỹ năng phù hợp:** {skill_match}\n"
            f"**Kỹ năng còn thiếu:** {skill_miss}\n"
            f"\n_Để xem chi tiết CV đầy đủ, HR có thể vào phần **View CV** trên hệ thống._"
        )
    except Exception as e:
        return f"Lỗi khi lấy thông tin ứng viên: {str(e)}"


@tool
async def get_cv_summary(position_id: int, mode: str) -> str:
    """
    Lấy thống kê tổng quan về số lượng CV và kết quả chấm điểm cho một vị trí tuyển dụng.
    Gọi khi HR hỏi về số lượng CV, tỷ lệ pass/fail, hoặc phân bổ điểm số.

    Args:
        position_id: ID của vị trí cần xem thống kê (tự động inject từ session)
    """
    try:
        data   = await recruitment_api.get_cv_statistics(position_id=position_id, mode=mode)
        total  = data.get("total", 0)
        scored = data.get("scored", 0)
        passed = data.get("passed", 0)
        failed = scored - passed
        return (
            f"**Thống kê CV — Vị trí ID {position_id}:**\n"
            f"- Tổng CV: {total}\n"
            f"- Đã chấm điểm: {scored}\n"
            f"- Pass (≥75đ): {passed}\n"
            f"- Fail (<75đ): {failed}"
        )
    except Exception as e:
        return f"Lỗi khi lấy thống kê CV: {str(e)}"


@tool
async def send_interview_email(
    app_cv_id: int,
    candidate_id: str,
    candidate_email: str,
    candidate_name: str,
    position_id: int,
    position_name: str,
    email_type: str,
    custom_message: str,
    interview_date: Optional[str] = None
) -> str:
    """
    Gửi email thông báo cho ứng viên qua SMTP.
    Hỗ trợ 3 loại: INTERVIEW_INVITE, OFFER_LETTER, REJECTION.

    LƯU Ý: Tool này KHÔNG được gọi trực tiếp bởi LLM.
    hr_graph sẽ intercept và yêu cầu HR xác nhận trước khi thực thi.

    Args:
        app_cv_id:       ID của Application CV (tự động resolve từ sql_metadata)
        candidate_id:    UUID của ứng viên (tự động resolve từ sql_metadata)
        candidate_email: Địa chỉ email của ứng viên (tự động resolve)
        candidate_name:  Tên ứng viên (tự động resolve)
        position_id:     ID vị trí ứng tuyển (tự động inject từ session)
        position_name:   Tên vị trí hiển thị trong email
        email_type:      INTERVIEW_INVITE | OFFER_LETTER | REJECTION
        custom_message:  Nội dung tùy chỉnh thêm vào email
        interview_date:  Ngày phỏng vấn ISO format (chỉ dùng cho INTERVIEW_INVITE)
    """
    valid_types = {"INTERVIEW_INVITE", "OFFER_LETTER", "REJECTION"}
    if email_type.upper() not in valid_types:
        return f"Loại email không hợp lệ: {email_type}. Chọn một trong: {valid_types}"

    try:
        await recruitment_api.send_interview_email(
            app_cv_id=app_cv_id,
            candidate_id=candidate_id,
            candidate_email=candidate_email,
            candidate_name=candidate_name,
            position_id=position_id,
            position_name=position_name,
            email_type=email_type.upper(),
            interview_date=interview_date,
            custom_message=custom_message
        )
        type_label = {
            "INTERVIEW_INVITE": "mời phỏng vấn",
            "OFFER_LETTER":     "offer letter",
            "REJECTION":        "từ chối",
        }.get(email_type.upper(), email_type)
        return f"Đã gửi email {type_label} thành công tới {candidate_name} ({candidate_email})."
    except Exception as e:
        return f"Lỗi khi gửi email: {str(e)}"


@tool
async def search_candidates_by_criteria(
    position_id: int,
    mode: str,
    min_score: Optional[int] = None,
    skill_keyword: Optional[str] = None,
    name_keyword: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Tìm kiếm và lọc ứng viên theo tiêu chí: điểm tối thiểu, kỹ năng, hoặc tên.
    Kết quả sắp xếp theo điểm giảm dần. Hỗ trợ lấy top N ứng viên cao nhất.
    Gọi khi HR muốn lọc danh sách hoặc xem top ứng viên.

    Args:
        position_id:   ID vị trí cần lọc (tự động inject từ session)
        mode:          Chế độ HR_MODE hoặc CANDIDATE_MODE (tự động inject từ session)
        min_score:     Lọc ứng viên có điểm >= giá trị này (optional)
        skill_keyword: Từ khóa kỹ năng tìm trong skillMatch (optional, không phân biệt hoa thường)
        name_keyword:  Từ khóa tên ứng viên (optional, không phân biệt hoa thường)
        top_k:         Chỉ trả về N ứng viên có điểm cao nhất (optional, dùng khi HR hỏi "top 3")
    """
    try:
        applications = await recruitment_api.get_applications(position_id=position_id)
        if not applications:
            return f"Không có ứng viên nào trong vị trí ID {position_id}."

        target_source = "HR" if mode == "HR_MODE" else "CANDIDATE"
        results = [app for app in applications if app.get("sourceType") == target_source]

        if min_score is not None:
            results = [app for app in results if (app.get("score") or 0) >= min_score]

        if skill_keyword:
            kw = skill_keyword.lower()
            results = [
                app for app in results
                if kw in (app.get("skillMatch") or "").lower()
            ]

        if name_keyword:
            kw = name_keyword.lower()
            results = [
                app for app in results
                if kw in (app.get("candidateName") or "").lower()
            ]

        # Sort by score descending (None scores go to the bottom)
        results.sort(key=lambda a: (a.get("score") or -1), reverse=True)

        if top_k is not None:
            results = results[:top_k]

        if not results:
            parts = []
            if min_score is not None:
                parts.append(f"điểm >= {min_score}")
            if skill_keyword:
                parts.append(f"kỹ năng '{skill_keyword}'")
            if name_keyword:
                parts.append(f"tên '{name_keyword}'")
            return f"Không tìm thấy ứng viên nào với {', '.join(parts) or 'tiêu chí đã cho'}."

        label = f"top {top_k}" if top_k else str(len(results))
        lines = [f"**Tìm thấy {label} ứng viên:**"]
        for i, app in enumerate(results, 1):
            name  = app.get("candidateName", "N/A")
            score = app.get("score", "Chưa chấm")
            stage = app.get("recruitmentStage", "APPLIED")
            lines.append(f"{i}. **{name}** — Điểm: {score}/100 — Trạng thái: {stage}")

        return "\n".join(lines)
    except Exception as e:
        return f"Lỗi khi tìm kiếm ứng viên: {str(e)}"

@tool
async def evaluate_candidates(
    candidate_names: list[str],
    position_id: int
) -> str:
    """
    Kích hoạt việc chấm điểm CV của một hoặc nhiều ứng viên.
    Gọi khi HR yêu cầu "hãy đánh giá ứng viên này", "chấm điểm các CV này".
    Lưu ý: Tool này KHÔNG được gọi trực tiếp bởi LLM mà sẽ được Graph intercept để chạy node chấm điểm `hr_scoring_node`.

    Args:
        candidate_names: Danh sách tên (hoặc ID) ứng viên cần chấm.
        position_id: ID vị trí (auto-injected).
    """
    # Graph sẽ chặn tool này và tự động gọi hr_scoring_node thay vì chạy logic trong này.
    return "Đang tiến hành chấm điểm ứng viên... Xin chờ trong giây lát."

# Registered tool list — order matters for bind_tools()
HR_TOOLS = [get_candidate_details, get_cv_summary, send_interview_email, search_candidates_by_criteria, evaluate_candidates]
