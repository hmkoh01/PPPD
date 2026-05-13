"""
이슈(Issue) API.

POST   /api/issues/{issue_id}/closeup   클로즈업 업로드 + Gemini 분석
PATCH  /api/issues/{issue_id}/retake    재촬영 초기화 (red 리셋)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
    summary="클로즈업 사진 업로드 + Gemini VLM 분석 (Step 3)",
)
async def upload_closeup(
    issue_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Any:
    """
    클로즈업 이미지를 저장하고 Gemini로 분석합니다.

    GeminiConfigError (API 키 미설정) 포함 모든 오류는 HTTP 200 + status=orange 로 처리합니다.
    (시연 안정성 정책 — README Gemini Service 섹션 참조)
    """
    issue = get_issue_or_404(db, issue_id)

    stored = await save_upload_file(
        file,
        ImageType.CLOSEUP,
        inspection_id=issue.inspection_id,
        issue_id=issue_id,
    )

    issue.closeup_image_path = stored.filename
    db.flush()

    # Gemini 분석 — 모든 실패는 suspicious fallback
    try:
        result = await analyze_closeup_async(image_path=stored.filename)
    except GeminiConfigError as exc:
        logger.warning("Gemini API 키 미설정, orange fallback 적용: %s", exc)
        result_str = "suspicious"
        reason_str = "AI 설정 오류로 관리자 확인이 필요합니다."
    except Exception as exc:
        logger.warning("Gemini 분석 예외 처리, orange fallback 적용: %s", exc)
        result_str = "suspicious"
        reason_str = "AI 분석 중 오류가 발생하여 관리자 확인이 필요합니다."
    else:
        result_str = result.result
        reason_str = result.reason

    issue.status = IssueStatus.GREEN if result_str == "clean" else IssueStatus.ORANGE
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
    summary="이슈 재촬영 초기화 — red 상태로 리셋",
)
def retake_issue(issue_id: int, db: Session = Depends(get_db)) -> Any:
    issue = get_issue_or_404(db, issue_id)
    issue.status = IssueStatus.RED
    issue.closeup_image_path = None
    issue.vlm_reason = None
    db.commit()
    db.refresh(issue)
    return issue_to_schema(issue)
