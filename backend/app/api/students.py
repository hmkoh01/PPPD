"""
학생(Student) 인증 API.

POST /api/students/verify  — 학번+이름으로 배정 호실 및 진행 중인 inspection 반환
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.student import Student
from app.schemas.inspection import InspectionOut
from app.schemas.room import RoomOut
from app.schemas.student import StudentOut
from app.services.inspection_service import (
    get_latest_inspection_for_student,
    inspection_to_schema,
    room_to_schema,
    student_to_schema,
)

router = APIRouter()


class StudentVerifyRequest(BaseModel):
    student_number: str
    name: str


class StudentVerifyResponse(BaseModel):
    student: StudentOut
    room: RoomOut
    inspection: InspectionOut | None = None
    status: str


@router.post(
    "/verify",
    response_model=StudentVerifyResponse,
    summary="학생 인증 — 학번+이름으로 배정 호실 및 점검 정보 반환",
)
def verify_student(
    body: StudentVerifyRequest,
    db: Session = Depends(get_db),
) -> Any:
    """
    학번(student_number)과 이름(name)이 일치하는 학생을 조회하고,
    배정된 호실과 가장 최신 inspection 정보를 반환합니다.

    Streamlit Step 0 인증 로직에 대응합니다.
    """
    student = (
        db.query(Student)
        .filter(
            Student.student_number == body.student_number,
            Student.name == body.name,
        )
        .first()
    )
    if student is None:
        raise HTTPException(
            status_code=404,
            detail="일치하는 배정 정보가 없습니다. 학번과 이름을 확인해 주세요.",
        )
    if student.room_id is None or student.room is None:
        raise HTTPException(
            status_code=404,
            detail="배정된 호실 정보가 없습니다. 관리자에게 문의하세요.",
        )

    inspection = get_latest_inspection_for_student(db, student.id)

    return StudentVerifyResponse(
        student=student_to_schema(student),
        room=room_to_schema(student.room),
        inspection=inspection_to_schema(inspection) if inspection else None,
        status=inspection.status if inspection else student.room.status,
    )
