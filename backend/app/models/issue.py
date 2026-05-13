"""
Issue ORM 모델.

CV 파이프라인(detect_difference)이 검출한 이상 영역 1건을 나타냅니다.

박스 좌표:
- 기존 Streamlit 의 box_coords=[x, y, w, h] JSON 을
  x, y, width, height 4개 정수 컬럼으로 분리하여 저장합니다.
- 하위 호환을 위해 box_coords property 를 제공합니다.

이미지:
- crop_image_path    : 최종 사진에서 박스 영역을 크롭한 이미지 (Step 3 썸네일용)
- closeup_image_path : 학생이 직접 촬영한 클로즈업 사진 (Gemini 분석 대상)
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
        default=IssueStatus.RED,
        comment="red=미검증 / orange=의심 / green=이상없음",
    )

    # ── 이미지 경로 ─────────────────────────────────────────────────────
    crop_image_path    = Column(String, nullable=True, comment="최종 사진 크롭 파일명")
    closeup_image_path = Column(String, nullable=True, comment="학생 촬영 클로즈업 파일명")

    # ── Gemini 판독 결과 ────────────────────────────────────────────────
    vlm_reason = Column(Text, nullable=True, comment="Gemini 한국어 판독 사유")

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
