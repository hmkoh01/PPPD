"""
관리자(Admin) 전용 API.

GET    /api/admin/rooms
POST   /api/admin/rooms
GET    /api/admin/rooms/{room_id}
GET    /api/admin/inspections
GET    /api/admin/inspections/{inspection_id}
PATCH  /api/admin/inspections/{inspection_id}/approve
PATCH  /api/admin/inspections/{inspection_id}/reject
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.constants import ImageType
from app.db.database import get_db
from app.models.inspection import Inspection
from app.models.room import Room
from app.schemas.inspection import InspectionOut
from app.schemas.room import AdminCreateRoomResponse, RoomDetailOut, RoomOut
from app.services.inspection_service import (
    approve_inspection,
    create_room_with_student_and_inspection,
    get_inspection_or_404,
    get_room_or_404,
    inspection_to_schema,
    reject_inspection,
    room_detail_to_schema,
    room_to_schema,
    student_to_schema,
)
from app.services.storage_service import get_public_image_url, save_upload_file

router = APIRouter()


# ────────────────────────────────────────────────────────────
# 호실 관리
# ────────────────────────────────────────────────────────────

@router.get(
    "/rooms",
    response_model=list[RoomDetailOut],
    summary="전체 호실 목록 조회 (학생·최신 inspection 포함)",
)
def list_rooms(db: Session = Depends(get_db)) -> Any:
    rooms = db.query(Room).order_by(Room.id).all()
    return [room_detail_to_schema(r) for r in rooms]


@router.post(
    "/rooms",
    response_model=AdminCreateRoomResponse,
    status_code=status.HTTP_201_CREATED,
    summary="호실 등록 — 학생 배정 + 기준 사진 업로드 (multipart)",
)
async def create_room(
    room_number: str = Form(..., description="호실 번호 (예: 101)"),
    student_number: str = Form(..., description="학번"),
    student_name: str = Form(..., description="학생 이름"),
    ref_image: UploadFile = File(..., description="기준 사진 파일"),
    db: Session = Depends(get_db),
) -> Any:
    """
    호실을 생성하고 학생을 배정하며, 기준 사진을 저장합니다.
    이 단계에서 Inspection(status=ready)도 함께 생성됩니다.
    """
    # 기준 사진 먼저 저장
    stored = await save_upload_file(
        ref_image,
        ImageType.REF,
        room_number=room_number,
    )

    try:
        room, student, inspection = create_room_with_student_and_inspection(
            db=db,
            room_number=room_number,
            student_number=student_number,
            student_name=student_name,
            ref_image_path=stored.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return AdminCreateRoomResponse(
        room=room_to_schema(room),
        student=student_to_schema(student),
        inspection=inspection_to_schema(inspection),
        ref_image_url=stored.public_url,
    )


@router.get(
    "/rooms/{room_id}",
    response_model=RoomDetailOut,
    summary="호실 단건 조회 (student + 최신 inspection 포함)",
)
def get_room(room_id: int, db: Session = Depends(get_db)) -> Any:
    room = get_room_or_404(db, room_id)
    return room_detail_to_schema(room)


# ────────────────────────────────────────────────────────────
# 점검 관리
# ────────────────────────────────────────────────────────────

@router.get(
    "/inspections",
    response_model=list[InspectionOut],
    summary="점검 목록 조회 (status 필터 지원)",
)
def list_inspections(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
) -> Any:
    """
    query param 예시: ?status_filter=pending_review
    생략하면 전체 목록을 최신순으로 반환합니다.
    """
    query = db.query(Inspection).order_by(Inspection.id.desc())
    if status_filter:
        query = query.filter(Inspection.status == status_filter)
    inspections = query.all()
    return [inspection_to_schema(i) for i in inspections]


@router.get(
    "/inspections/{inspection_id}",
    response_model=InspectionOut,
    summary="점검 상세 조회 (issues + 모든 image_url 포함)",
)
def get_inspection(inspection_id: int, db: Session = Depends(get_db)) -> Any:
    insp = get_inspection_or_404(db, inspection_id)
    return inspection_to_schema(insp)


class ApproveRequest(BaseModel):
    admin_feedback: Optional[str] = None


class RejectRequest(BaseModel):
    admin_feedback: str


@router.patch(
    "/inspections/{inspection_id}/approve",
    response_model=InspectionOut,
    summary="점검 승인",
)
def approve(
    inspection_id: int,
    body: ApproveRequest,
    db: Session = Depends(get_db),
) -> Any:
    insp = get_inspection_or_404(db, inspection_id)
    insp = approve_inspection(db, insp, admin_feedback=body.admin_feedback)
    return inspection_to_schema(insp)


@router.patch(
    "/inspections/{inspection_id}/reject",
    response_model=InspectionOut,
    summary="점검 반려 (admin_feedback 필수)",
)
def reject(
    inspection_id: int,
    body: RejectRequest,
    db: Session = Depends(get_db),
) -> Any:
    if not body.admin_feedback or not body.admin_feedback.strip():
        raise HTTPException(
            status_code=400,
            detail="반려 시 admin_feedback 은 필수입니다.",
        )
    insp = get_inspection_or_404(db, inspection_id)
    insp = reject_inspection(db, insp, admin_feedback=body.admin_feedback.strip())
    return inspection_to_schema(insp)
