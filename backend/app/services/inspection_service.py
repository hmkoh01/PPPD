"""
기숙사 퇴사 점검 — inspection 비즈니스 로직 서비스.

반복 사용되는 ORM→스키마 변환 헬퍼와 핵심 비즈니스 로직을 분리합니다.
API 라우터(admin.py, inspections.py, issues.py)에서 import 하여 사용합니다.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.inspection import Inspection
from app.models.issue import Issue
from app.models.room import Room
from app.models.student import Student
from app.schemas.inspection import InspectionOut, InspectionStatusOut
from app.schemas.issue import IssueOut
from app.schemas.room import RoomDetailOut, RoomOut
from app.schemas.student import StudentOut
from app.services.storage_service import get_public_image_url


# ────────────────────────────────────────────────────────────
# ORM → 스키마 변환 헬퍼
# ────────────────────────────────────────────────────────────

def _url(path: str | None) -> str | None:
    """파일명 또는 None을 public URL 또는 None으로 변환합니다."""
    return get_public_image_url(path) if path else None


def issue_to_schema(issue: Issue) -> IssueOut:
    return IssueOut(
        id=issue.id,
        inspection_id=issue.inspection_id,
        x=issue.x,
        y=issue.y,
        width=issue.width,
        height=issue.height,
        status=issue.status,
        candidate_type=issue.candidate_type or "small_damage",
        student_note=issue.student_note,
        vlm_reason=issue.vlm_reason,
        crop_image_path=issue.crop_image_path,
        crop_image_url=_url(issue.crop_image_path),
        closeup_image_path=issue.closeup_image_path,
        closeup_image_url=_url(issue.closeup_image_path),
    )


def inspection_to_schema(inspection: Inspection) -> InspectionOut:
    return InspectionOut(
        id=inspection.id,
        room_id=inspection.room_id,
        student_id=inspection.student_id,
        status=inspection.status,
        admin_feedback=inspection.admin_feedback,
        submitted_at=inspection.submitted_at,
        reviewed_at=inspection.reviewed_at,
        ref_image_path=inspection.ref_image_path,
        ref_image_url=_url(inspection.ref_image_path),
        initial_image_path=inspection.initial_image_path,
        initial_image_url=_url(inspection.initial_image_path),
        final_image_path=inspection.final_image_path,
        final_image_url=_url(inspection.final_image_path),
        issues=[issue_to_schema(i) for i in inspection.issues],
    )


def inspection_status_schema(inspection: Inspection) -> InspectionStatusOut:
    return InspectionStatusOut(
        inspection_id=inspection.id,
        status=inspection.status,
        admin_feedback=inspection.admin_feedback,
    )


def student_to_schema(student: Student) -> StudentOut:
    return StudentOut(
        id=student.id,
        student_number=student.student_number,
        name=student.name,
        room_id=student.room_id,
    )


def room_to_schema(room: Room) -> RoomOut:
    return RoomOut(id=room.id, room_number=room.room_number, status=room.status)


def room_detail_to_schema(room: Room) -> RoomDetailOut:
    """Room + student + 최신 inspection 포함 상세 스키마."""
    student = room.students[0] if room.students else None
    latest = room.inspections[-1] if room.inspections else None
    ref_url = _url(latest.ref_image_path) if latest else None
    return RoomDetailOut(
        id=room.id,
        room_number=room.room_number,
        status=room.status,
        student=student_to_schema(student) if student else None,
        latest_inspection=inspection_to_schema(latest) if latest else None,
        ref_image_url=ref_url,
    )


# ────────────────────────────────────────────────────────────
# 공통 조회 헬퍼
# ────────────────────────────────────────────────────────────

def get_inspection_or_404(db: Session, inspection_id: int) -> Inspection:
    insp = db.get(Inspection, inspection_id)
    if insp is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="점검 세션을 찾을 수 없습니다.")
    return insp


def get_room_or_404(db: Session, room_id: int) -> Room:
    room = db.get(Room, room_id)
    if room is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="호실을 찾을 수 없습니다.")
    return room


def get_issue_or_404(db: Session, issue_id: int) -> Issue:
    issue = db.get(Issue, issue_id)
    if issue is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="이슈를 찾을 수 없습니다.")
    return issue


# ────────────────────────────────────────────────────────────
# 비즈니스 로직
# ────────────────────────────────────────────────────────────

def get_latest_inspection_for_student(
    db: Session, student_id: int
) -> Inspection | None:
    """해당 학생의 가장 최신 Inspection을 반환합니다."""
    return (
        db.query(Inspection)
        .filter(Inspection.student_id == student_id)
        .order_by(Inspection.id.desc())
        .first()
    )


def create_room_with_student_and_inspection(
    db: Session,
    room_number: str,
    student_number: str,
    student_name: str,
    ref_image_path: str,
) -> tuple[Room, Student, Inspection]:
    """
    호실 + 학생 + inspection을 트랜잭션 안에서 한 번에 생성합니다.
    room_number 중복 시 ValueError를 발생시킵니다.
    """
    from app.core.constants import InspectionStatus, RoomStatus

    existing = db.query(Room).filter(Room.room_number == room_number).first()
    if existing is not None:
        raise ValueError(f"호실 번호 '{room_number}' 가 이미 존재합니다.")

    room = Room(room_number=room_number, status=RoomStatus.READY)
    db.add(room)
    db.flush()

    student = Student(
        student_number=student_number,
        name=student_name,
        room_id=room.id,
    )
    db.add(student)
    db.flush()

    inspection = Inspection(
        room_id=room.id,
        student_id=student.id,
        status=InspectionStatus.READY,
        ref_image_path=ref_image_path,
    )
    db.add(inspection)
    db.commit()
    db.refresh(room)
    db.refresh(student)
    db.refresh(inspection)
    return room, student, inspection


def approve_inspection(
    db: Session,
    inspection: Inspection,
    admin_feedback: str | None = None,
) -> Inspection:
    from app.core.constants import InspectionStatus, RoomStatus

    inspection.status = InspectionStatus.APPROVED
    inspection.reviewed_at = datetime.utcnow()
    if admin_feedback:
        inspection.admin_feedback = admin_feedback
    room = db.get(Room, inspection.room_id)
    if room:
        room.status = RoomStatus.APPROVED
    db.commit()
    db.refresh(inspection)
    return inspection


def reject_inspection(
    db: Session,
    inspection: Inspection,
    admin_feedback: str,
) -> Inspection:
    from app.core.constants import InspectionStatus, RoomStatus

    inspection.status = InspectionStatus.REJECTED
    inspection.reviewed_at = datetime.utcnow()
    inspection.admin_feedback = admin_feedback
    room = db.get(Room, inspection.room_id)
    if room:
        room.status = RoomStatus.REJECTED
    db.commit()
    db.refresh(inspection)
    return inspection
