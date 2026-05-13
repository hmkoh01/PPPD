"""
도메인 상태값 상수 모음.

SQLAlchemy 모델의 status 컬럼에 저장되는 문자열 값을 한 곳에서 관리합니다.
Enum 대신 클래스 상수를 사용하여 DB 마이그레이션 없이 자유롭게 확장할 수 있습니다.
"""
from __future__ import annotations


class RoomStatus:
    """
    rooms.status 에 저장되는 값.
    최신 Inspection 상태의 비정규화(denormalized) 캐시 역할을 겸합니다.
    """
    READY          = "ready"           # 호실 등록 완료, 점검 미시작
    CHECKED_IN     = "checked_in"      # 학생 초기 사진 촬영 완료
    PENDING_REVIEW = "pending_review"  # 학생 최종 제출 완료, 관리자 검토 대기
    APPROVED       = "approved"        # 관리자 합격 처리
    REJECTED       = "rejected"        # 관리자 재점검 요청

    ALL: list[str] = [READY, CHECKED_IN, PENDING_REVIEW, APPROVED, REJECTED]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.ALL


class InspectionStatus:
    """inspections.status 에 저장되는 값."""
    READY          = "ready"
    CHECKED_IN     = "checked_in"
    PENDING_REVIEW = "pending_review"
    APPROVED       = "approved"
    REJECTED       = "rejected"

    ALL: list[str] = [READY, CHECKED_IN, PENDING_REVIEW, APPROVED, REJECTED]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.ALL


class IssueStatus:
    """issues.status 에 저장되는 값."""
    RED    = "red"     # 미검증 (클로즈업 미촬영)
    ORANGE = "orange"  # Gemini → suspicious (의심 영역)
    GREEN  = "green"   # Gemini → clean (이상 없음)

    ALL: list[str] = [RED, ORANGE, GREEN]

    # 제출 가능 상태 (red 가 하나라도 남아 있으면 제출 불가)
    RESOLVED: list[str] = [ORANGE, GREEN]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.ALL

    @classmethod
    def is_resolved(cls, value: str) -> bool:
        return value in cls.RESOLVED


class ImageType:
    """
    storage_service 에서 파일명 prefix 에 사용하는 이미지 타입 상수.

    사용법:
        from app.core.constants import ImageType
        stored = save_ndarray(img, ImageType.INITIAL, room_number="101")
    """
    REF           = "ref"           # 관리자 등록 기준 사진
    INITIAL       = "initial"       # 학생 Step 1 입사 초기 사진
    FINAL         = "final"         # 학생 Step 2 퇴사 최종 사진 (정합 전 원본)
    ALIGNED_FINAL = "aligned_final" # ORB 정합 완료된 최종 사진
    CROP          = "crop"          # 이슈 박스 크롭 (자동 생성, 썸네일용)
    CLOSEUP       = "closeup"       # 학생 Step 3 클로즈업 촬영 사진

    ALL: list[str] = [REF, INITIAL, FINAL, ALIGNED_FINAL, CROP, CLOSEUP]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.ALL
