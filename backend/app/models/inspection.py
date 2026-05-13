"""
Inspection ORM 모델.

한 번의 퇴사 점검 세션을 나타냅니다.

MVP 설계:
- 학생의 "현재 진행 중인 점검"은 해당 student_id 기준 가장 최근(created_at DESC) Inspection으로 조회합니다.
- 재점검 이력 고도화(별도 attempt 번호, parent 링크 등)는 TODO 로 남깁니다.

이미지 경로:
- ref_image_path     : 관리자가 등록한 기준 사진 (점검 시작 시 Room 에서 복사)
- initial_image_path : Step 1 에서 학생이 촬영한 입사 초기 사진
- final_image_path   : Step 2 에서 촬영 후 ORB 정합된 최종 사진
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.constants import InspectionStatus
from app.db.database import Base, TimestampMixin


class Inspection(TimestampMixin, Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)

    room_id = Column(
        Integer,
        ForeignKey("rooms.id", ondelete="CASCADE"),
        nullable=False,
        comment="소속 호실의 내부 ID",
    )
    student_id = Column(
        Integer,
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
        comment="점검을 수행한 학생의 내부 ID",
    )

    # ── 상태 ────────────────────────────────────────────────────────────
    status = Column(
        String,
        nullable=False,
        default=InspectionStatus.READY,
        comment="점검 진행 상태",
    )

    # ── 이미지 경로 (상대 파일명, StaticFiles /images/ 로 서빙) ─────────
    ref_image_path     = Column(String, nullable=True, comment="기준 사진 파일명")
    initial_image_path = Column(String, nullable=True, comment="입사 초기 사진 파일명")
    final_image_path   = Column(String, nullable=True, comment="정합된 퇴사 최종 사진 파일명")

    # ── 관리자 판정 ─────────────────────────────────────────────────────
    admin_feedback = Column(Text, nullable=True, comment="관리자 피드백 텍스트")

    # ── 이벤트 타임스탬프 (created_at/updated_at 은 TimestampMixin 에서 상속) ──
    submitted_at = Column(
        DateTime,
        nullable=True,
        comment="학생이 최종 제출한 시각 (status → pending_review)",
    )
    reviewed_at = Column(
        DateTime,
        nullable=True,
        comment="관리자가 판정한 시각 (status → approved/rejected)",
    )

    # TODO: 재점검 이력 고도화
    #   - attempt_number = Column(Integer, default=1)
    #   - parent_inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=True)

    # ── Relationships ────────────────────────────────────────────────────
    room = relationship("Room", back_populates="inspections")
    student = relationship("Student", back_populates="inspections")
    issues = relationship(
        "Issue",
        back_populates="inspection",
        cascade="all, delete-orphan",
        order_by="Issue.id",
    )

    def __repr__(self) -> str:
        return (
            f"<Inspection id={self.id} room_id={self.room_id}"
            f" student_id={self.student_id} status={self.status!r}>"
        )
