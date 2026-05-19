"""
점검 세션(Inspection) API.

GET    /api/inspections/{inspection_id}
GET    /api/inspections/{inspection_id}/status
POST   /api/inspections/{inspection_id}/initial-image
POST   /api/inspections/{inspection_id}/final-image
POST   /api/inspections/{inspection_id}/submit
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.constants import ImageType, InspectionStatus, IssueStatus, RoomStatus
from app.db.database import get_db
from app.models.issue import Issue
from app.models.room import Room
from app.schemas.inspection import (
    FinalImageResponse,
    InitialImageResponse,
    InspectionOut,
    InspectionStatusOut,
)
from app.schemas.issue import IssueOut
from app.services.inspection_service import (
    get_inspection_or_404,
    inspection_to_schema,
    inspection_status_schema,
    issue_to_schema,
)
from app.services.storage_service import get_public_image_url, save_upload_file
from app.services.vision_service import (
    VisionAlignmentFailed,
    VisionDetectionFailed,
    VisionServiceError,
    align_and_detect,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/{inspection_id}",
    response_model=InspectionOut,
    summary="점검 상세 조회 (issues + 모든 image_url 포함)",
)
def get_inspection(inspection_id: int, db: Session = Depends(get_db)) -> Any:
    insp = get_inspection_or_404(db, inspection_id)
    return inspection_to_schema(insp)


@router.get(
    "/{inspection_id}/status",
    response_model=InspectionStatusOut,
    summary="점검 상태 조회 (Step 4 대기 화면용 최소 정보)",
)
def get_inspection_status(inspection_id: int, db: Session = Depends(get_db)) -> Any:
    insp = get_inspection_or_404(db, inspection_id)
    return inspection_status_schema(insp)


@router.post(
    "/{inspection_id}/initial-image",
    response_model=InitialImageResponse,
    summary="입사 초기 사진 업로드 (Step 1)",
)
async def upload_initial_image(
    inspection_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Any:
    insp = get_inspection_or_404(db, inspection_id)
    room = db.get(Room, insp.room_id)

    stored = await save_upload_file(
        file,
        ImageType.INITIAL,
        room_number=room.room_number if room else None,
        inspection_id=inspection_id,
    )

    insp.initial_image_path = stored.filename
    insp.status = InspectionStatus.CHECKED_IN
    if room:
        room.status = RoomStatus.CHECKED_IN
    db.commit()
    db.refresh(insp)

    return InitialImageResponse(
        inspection_id=insp.id,
        status=insp.status,
        initial_image_path=stored.filename,
        initial_image_url=stored.public_url,
    )


@router.post(
    "/{inspection_id}/final-image",
    response_model=FinalImageResponse,
    summary="퇴사 최종 사진 업로드 + CV 분석 (Step 2)",
)
async def upload_final_image(
    inspection_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Any:
    """
    final 이미지를 업로드하고 즉시 CV 분석을 실행합니다.

    - VisionAlignmentFailed → HTTP 422 (같은 위치·구도로 재촬영 안내)
    - VisionDetectionFailed / VisionServiceError → HTTP 422
    - 기타 예외 → HTTP 500

    MVP: aligned final 이미지를 final_image_path 에 저장합니다.
    기존 이슈는 모두 삭제하고 새로 생성합니다.
    """
    insp = get_inspection_or_404(db, inspection_id)

    if not insp.initial_image_path:
        raise HTTPException(
            status_code=400,
            detail="초기 사진이 아직 업로드되지 않았습니다. Step 1을 먼저 진행해 주세요.",
        )

    room = db.get(Room, insp.room_id)
    room_number = room.room_number if room else None

    # final 이미지 bytes 읽기 (vision_service 에 전달)
    image_bytes = await file.read()

    # CV 분석 실행
    try:
        vision_result = align_and_detect(
            initial_image_path=insp.initial_image_path,
            final_image_bytes=image_bytes,
            room_number=room_number,
            inspection_id=inspection_id,
        )
    except VisionAlignmentFailed as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"이미지 정합 실패 — 같은 위치와 구도로 다시 촬영해 주세요. ({exc})",
        ) from exc
    except (VisionDetectionFailed, VisionServiceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CV 분석 실패 — {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("final-image CV 처리 중 예기치 않은 오류")
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류가 발생했습니다: {exc}",
        ) from exc

    # 기존 이슈 삭제 (재촬영 시 초기화)
    db.query(Issue).filter(Issue.inspection_id == inspection_id).delete()

    # inspection 업데이트
    insp.final_image_path = vision_result.aligned_final_image_path

    # 새 이슈 생성 — AI 감지 결과는 "확인 필요 후보"로 등록 (손상 확정이 아님)
    new_issues: list[Issue] = []
    for di in vision_result.issues:
        issue = Issue(
            inspection_id=inspection_id,
            x=di.x,
            y=di.y,
            width=di.width,
            height=di.height,
            status=IssueStatus.NEEDS_CONFIRMATION,
            crop_image_path=di.crop_image_path,
            candidate_type=di.candidate_type,
        )
        db.add(issue)
        new_issues.append(issue)

    # 확인 필요 영역이 감지된 경우 inspection/room 상태를 needs_confirmation 으로 전환
    if new_issues:
        insp.status = InspectionStatus.NEEDS_CONFIRMATION
        if room:
            room.status = RoomStatus.NEEDS_CONFIRMATION

    db.commit()
    db.refresh(insp)
    for i in new_issues:
        db.refresh(i)

    return FinalImageResponse(
        inspection=inspection_to_schema(insp),
        final_image_url=get_public_image_url(insp.final_image_path) if insp.final_image_path else None,
        issues=[issue_to_schema(i) for i in new_issues],
    )


@router.post(
    "/{inspection_id}/submit",
    response_model=InspectionStatusOut,
    summary="최종 제출 → 관리자 검토 대기 (Step 3 완료)",
)
def submit_inspection(inspection_id: int, db: Session = Depends(get_db)) -> Any:
    """
    모든 issue 가 green 또는 orange 인지 확인한 후 pending_review 로 전환합니다.
    red 이슈가 남아 있으면 400을 반환합니다.
    이슈가 없으면 즉시 제출 가능합니다.
    """
    insp = get_inspection_or_404(db, inspection_id)

    # 확인 자료(근접 촬영+메모)가 제출되지 않은 이슈가 있으면 제출 불가
    unresolved = [i for i in insp.issues if i.status == IssueStatus.NEEDS_CONFIRMATION]
    if unresolved:
        raise HTTPException(
            status_code=400,
            detail=f"확인 자료가 제출되지 않은 영역 {len(unresolved)}건이 남아 있습니다. 모든 영역의 근접 촬영을 완료해 주세요.",
        )

    insp.status = InspectionStatus.PENDING_REVIEW
    insp.submitted_at = datetime.utcnow()
    room = db.get(Room, insp.room_id)
    if room:
        room.status = RoomStatus.PENDING_REVIEW
    db.commit()
    db.refresh(insp)

    return inspection_status_schema(insp)
