"""
이슈(Issue) API.

설계 철학: AI는 판정자가 아니라 확인 보조자입니다.
  AI가 감지한 "확인 필요 후보 영역"에 대해 학생이 근접 촬영과 메모를 제출하고,
  관리자가 모든 자료를 종합하여 최종 판단합니다.

POST   /api/issues/{issue_id}/closeup   근접 촬영 업로드 + 메모 저장 + Gemini 참고 분석
PATCH  /api/issues/{issue_id}/retake    재촬영 초기화 (needs_confirmation 으로 리셋)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.constants import ImageType, IssueStatus
from app.db.database import get_db
from app.schemas.issue import CloseupUploadResponse, IssueOut
from app.services.gemini_service import GeminiConfigError, analyze_closeup_async
from app.services.inspection_service import get_issue_or_404, issue_to_schema
from app.services.storage_service import get_public_image_url, save_upload_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{issue_id}/closeup",
    response_model=CloseupUploadResponse,
    summary="근접 확인 사진 업로드 + 학생 메모 저장 + Gemini VLM 참고 분석",
)
async def upload_closeup(
    issue_id: int,
    file: UploadFile = File(...),
    student_note: str | None = Form(None),
    db: Session = Depends(get_db),
) -> Any:
    """
    학생이 확인 필요 영역에 대해 근접 사진을 촬영하고 메모를 제출합니다.

    처리 순서:
      1. 근접 사진 저장
      2. 학생 메모 저장
      3. Gemini VLM 참고 분석 실행 (실패 시에도 자료 제출 처리)
      4. issue.status → evidence_submitted

    GeminiConfigError(API 키 미설정) 포함 모든 Gemini 오류는 fallback 처리합니다.
    Gemini 결과는 관리자 검토를 위한 "참고 의견"이며 판정 근거가 아닙니다.
    """
    issue = get_issue_or_404(db, issue_id)

    stored = await save_upload_file(
        file,
        ImageType.CLOSEUP,
        inspection_id=issue.inspection_id,
        issue_id=issue_id,
    )

    issue.closeup_image_path = stored.filename
    issue.student_note = student_note.strip() if student_note else None
    db.flush()

    # Gemini VLM 참고 분석 — student_note 를 함께 전달하여 이미지와 대조 분석
    # 모든 실패는 fallback 처리 (자료 제출 자체는 성공으로 처리)
    try:
        result = await analyze_closeup_async(
            image_path=stored.filename,
            student_note=issue.student_note,
        )
    except GeminiConfigError as exc:
        logger.warning("Gemini API 키 미설정, fallback 적용: %s", exc)
        result_str = "suspicious"
        reason_str = "AI 설정 오류로 관리자가 직접 확인해 주세요."
    except Exception as exc:
        logger.warning("Gemini 분석 예외, fallback 적용: %s", exc)
        result_str = "suspicious"
        reason_str = "AI 분석 중 오류가 발생했습니다. 관리자가 직접 확인해 주세요."
    else:
        result_str = result.result
        reason_str = result.reason

    # 학생이 근접 사진과 메모를 제출했으므로 evidence_submitted 로 전환
    issue.status = IssueStatus.EVIDENCE_SUBMITTED
    issue.vlm_reason = reason_str
    db.commit()
    db.refresh(issue)

    return CloseupUploadResponse(
        issue=issue_to_schema(issue),
        closeup_image_url=get_public_image_url(stored.filename),
        result=result_str,
        reason=reason_str,
    )


@router.patch(
    "/{issue_id}/retake",
    response_model=IssueOut,
    summary="확인 자료 재제출 초기화 — needs_confirmation 으로 리셋",
)
def retake_issue(issue_id: int, db: Session = Depends(get_db)) -> Any:
    """학생이 근접 촬영을 다시 하고 싶을 때 해당 이슈를 초기 상태로 되돌립니다."""
    issue = get_issue_or_404(db, issue_id)
    issue.status = IssueStatus.NEEDS_CONFIRMATION
    issue.closeup_image_path = None
    issue.student_note = None
    issue.vlm_reason = None
    db.commit()
    db.refresh(issue)
    return issue_to_schema(issue)
