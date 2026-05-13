"""
Student ORM 모델.

학생(점검 대상자) 정보를 저장합니다.
- student_number : 학번 (인증 키)
- name           : 학생 이름
- room_id        : 배정된 호실의 내부 정수 ID (rooms.id 참조)
"""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.database import Base, TimestampMixin


class Student(TimestampMixin, Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_number = Column(
        String,
        unique=True,
        nullable=False,
        comment="학번 (예: 2024123456). 인증 시 사용.",
    )
    name = Column(
        String,
        nullable=False,
        comment="학생 이름",
    )
    room_id = Column(
        Integer,
        ForeignKey("rooms.id", ondelete="SET NULL"),
        nullable=True,
        comment="배정된 호실의 내부 ID (rooms.id)",
    )

    # ── Relationships ────────────────────────────────────────────────────
    room = relationship(
        "Room",
        back_populates="students",
        foreign_keys=[room_id],
    )
    # 이 학생의 점검 이력 (재점검 포함, 생성 순 정렬)
    inspections = relationship(
        "Inspection",
        back_populates="student",
        order_by="Inspection.id",
    )

    def __repr__(self) -> str:
        return (
            f"<Student id={self.id} student_number={self.student_number!r}"
            f" name={self.name!r} room_id={self.room_id}>"
        )
