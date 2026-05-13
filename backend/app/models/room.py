"""
Room ORM 모델.

호실(방) 정보를 저장합니다.
- room_number : 관리자가 지정한 호실 번호 문자열 (Streamlit의 room_id 에 대응)
- id          : 내부 정수 PK (외래 키 참조용)
- status      : 최신 Inspection 상태의 비정규화 캐시 (빠른 조회용)
"""
from __future__ import annotations

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from app.core.constants import RoomStatus
from app.db.database import Base, TimestampMixin


class Room(TimestampMixin, Base):
    __tablename__ = "rooms"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="내부 정수 PK (외래 키 참조용)",
    )
    room_number = Column(
        String,
        unique=True,
        nullable=False,
        comment="관리자 지정 호실 번호 (예: 101, A-205). 구 Streamlit room_id에 대응.",
    )
    status = Column(
        String,
        nullable=False,
        default=RoomStatus.READY,
        comment="현재 상태 (최신 Inspection 상태의 비정규화 캐시)",
    )

    # ── Relationships ────────────────────────────────────────────────────
    # 이 호실에 배정된 학생 목록 (통상 1명)
    students = relationship(
        "Student",
        back_populates="room",
        foreign_keys="Student.room_id",
    )
    # 이 호실의 점검 이력 (재점검 포함, 생성 순 정렬)
    inspections = relationship(
        "Inspection",
        back_populates="room",
        cascade="all, delete-orphan",
        order_by="Inspection.id",
    )

    def __repr__(self) -> str:
        return f"<Room id={self.id} room_number={self.room_number!r} status={self.status!r}>"
