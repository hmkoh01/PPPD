"""
Issue ORM 모델.

CV 파이프라인(detect_difference)이 감지한 "확인 필요 후보 영역" 1건을 나타냅니다.

설계 철학: AI는 판정자가 아니라 확인 보조자입니다.
  - AI는 전후 차이 후보를 감지(NEEDS_CONFIRMATION)합니다.
  - 학생은 근접 촬영과 메모를 제출(EVIDENCE_SUBMITTED)합니다.
  - 관리자가 전후 사진, 학생 자료, VLM 참고 의견을 종합하여 최종 판단합니다.

박스 좌표:
- x, y, width, height 4개 정수 컬럼으로 분리하여 저장합니다.
- 하위 호환을 위해 box_coords property 를 제공합니다.

이미지:
- crop_image_path    : 최종 사진에서 박스 영역을 크롭한 이미지 (관리자 검토용 썸네일)
- closeup_image_path : 학생이 직접 촬영한 근접 확인 사진 (Gemini VLM 참고 분석 대상)

학생 자료:
- student_note : 학생이 근접 촬영 시 함께 남긴 메모 (선택 사항)
"""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.constants import IssueStatus
from app.db.database import Base, TimestampMixin


class Issue(TimestampMixin, Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(
        Integer,
        ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
        comment="소속 점검 세션 ID",
    )

    # ── 박스 좌표 (최종 정합 사진 기준 픽셀 좌표) ────────────────────────
    x      = Column(Integer, nullable=False, comment="박스 좌상단 x 좌표 (px)")
    y      = Column(Integer, nullable=False, comment="박스 좌상단 y 좌표 (px)")
    width  = Column(Integer, nullable=False, comment="박스 너비 (px)")
    height = Column(Integer, nullable=False, comment="박스 높이 (px)")

    # ── 상태 ────────────────────────────────────────────────────────────
    status = Column(
        String,
        nullable=False,
        default=IssueStatus.NEEDS_CONFIRMATION,
        comment="needs_confirmation=학생 자료 미제출 / evidence_submitted=자료 제출 완료 / cleared=확인 불필요",
    )

    # ── 이미지 경로 ─────────────────────────────────────────────────────
    crop_image_path    = Column(String, nullable=True, comment="최종 사진 크롭 파일명 (관리자 검토용)")
    closeup_image_path = Column(String, nullable=True, comment="학생 촬영 근접 확인 사진 파일명")

    # ── 학생 제출 자료 ───────────────────────────────────────────────────
    student_note = Column(Text, nullable=True, comment="학생이 근접 촬영 시 남긴 메모 (선택 사항)")

    # ── AI 감지 메타 ─────────────────────────────────────────────────────
    candidate_type = Column(
        String,
        nullable=True,
        default="small_damage",
        comment="small_damage=작은 손상 후보 / large_object=큰 객체/방치물 / recapture_recommended=재촬영 권장",
    )

    # ── Gemini VLM 참고 의견 ────────────────────────────────────────────
    vlm_reason = Column(Text, nullable=True, comment="Gemini VLM 참고 의견 (관리자 검토 참고용)")

    # ── Relationships ────────────────────────────────────────────────────
    inspection = relationship("Inspection", back_populates="issues")

    # ── 하위 호환 프로퍼티 ───────────────────────────────────────────────

    @property
    def box_coords(self) -> list[int]:
        """
        기존 Streamlit 코드와의 호환을 위해 [x, y, width, height] 형태로 반환합니다.

        Usage:
            x, y, bw, bh = issue.box_coords
        """
        return [self.x, self.y, self.width, self.height]

    def __repr__(self) -> str:
        return (
            f"<Issue id={self.id} inspection_id={self.inspection_id}"
            f" box=({self.x},{self.y},{self.width},{self.height}) status={self.status!r}>"
        )
