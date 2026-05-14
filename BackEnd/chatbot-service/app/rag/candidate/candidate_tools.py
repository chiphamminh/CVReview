from langchain_core.tools import tool
from typing import List, Optional
from app.services.recruitment_api import recruitment_api


@tool
async def evaluate_cv_fit(position_ids: List[int]) -> str:
    """
    Tính điểm phù hợp đa chiều của CV với một hoặc nhiều JDs.
    Gọi khi user hỏi về độ phù hợp hoặc tìm việc.

    Args:
        position_ids: Danh sách positionId cần chấm điểm (tối đa 10)
    """
    # Scoring is handled by the dedicated scoring_node. This tool is a marker
    # that signals intent when the scored_jobs cache is cold.
    return "Đang phân tích độ phù hợp của bạn với các vị trí. Vui lòng chờ..."


@tool
async def finalize_application(
    position_id: int,
    overall_status: str,
    technical_score: int,
    experience_score: int,
    ai_assessment: str,
    candidate_id: str,
    session_id: str,
) -> str:
    """
    Nộp đơn ứng tuyển chính thức. CHỈ được gọi khi điểm trung bình >= ngưỡng tối thiểu của vị trí.

    Args:
        position_id:      ID của vị trí ứng tuyển
        overall_status:   MatchStatus do AI sinh ra (chỉ để hiển thị)
        technical_score:  Điểm kỹ thuật (0-100) — tự động inject từ scored_jobs
        experience_score: Điểm kinh nghiệm (0-100) — tự động inject từ scored_jobs
        ai_assessment:    Nhận xét AI tổng hợp
        candidate_id:     UUID của ứng viên (tự động inject từ session)
        session_id:       ID phiên hội thoại (tự động inject từ session)
    """
    try:
        res = await recruitment_api.finalize_application(
            candidate_id=candidate_id,
            position_id=position_id,
            technical_score=technical_score,
            experience_score=experience_score,
            overall_status=overall_status,
            ai_assessment=ai_assessment,
            learning_path=None,
            session_id=session_id,
        )
        return f"Nộp đơn thành công. Application CV ID: {res.get('applicationCvId')}"
    except Exception as e:
        return f"Lỗi khi nộp đơn: {str(e)}"


@tool
async def check_application_status(
    candidate_id: str,
    position_id: Optional[int] = None,
) -> str:
    """
    Kiểm tra trạng thái ứng tuyển của ứng viên — đã nộp đơn vị trí nào, điểm bao nhiêu.
    Gọi khi candidate hỏi "Tôi đã apply chưa?", "Đơn của tôi thế nào?", "Tôi đã nộp vị trí nào?".

    Args:
        candidate_id: UUID của ứng viên (tự động inject từ session)
        position_id:  ID vị trí muốn kiểm tra cụ thể (optional — None = lấy tất cả)
    """
    try:
        data = await recruitment_api.check_application_status(
            candidate_id=candidate_id,
            position_id=position_id,
        )
        applications: list = data.get("applications", [])

        if not applications:
            if position_id:
                return f"Bạn chưa nộp đơn vào vị trí ID {position_id}."
            return "Bạn chưa nộp đơn vào vị trí nào."

        lines = ["Trạng thái ứng tuyển của bạn:"]
        for app in applications:
            pos_name = app.get("positionName", f"Position #{app.get('positionId')}")
            score    = app.get("score", "Chưa chấm")
            status   = app.get("status", "Đã nộp")
            lines.append(f"• {pos_name} — Điểm: {score} | Trạng thái: {status}")

        return "\n".join(lines)
    except Exception as e:
        return f"Lỗi khi kiểm tra trạng thái ứng tuyển: {str(e)}"


CANDIDATE_TOOLS = [evaluate_cv_fit, finalize_application, check_application_status]
