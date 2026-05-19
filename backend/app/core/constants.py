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
    READY                = "ready"                # 호실 등록 완료, 점검 미시작
    CHECKED_IN           = "checked_in"           # 학생 초기 사진 촬영 완료
    NEEDS_CONFIRMATION   = "needs_confirmation"   # AI 차이 후보 감지, 학생 확인 자료 대기 중
    PENDING_REVIEW       = "pending_review"       # 학생 최종 제출 완료, 관리자 검토 대기
    APPROVED             = "approved"             # 관리자 최종 승인
    REJECTED             = "rejected"             # 관리자 재점검 요청

    ALL: list[str] = [READY, CHECKED_IN, NEEDS_CONFIRMATION, PENDING_REVIEW, APPROVED, REJECTED]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.ALL


class InspectionStatus:
    """inspections.status 에 저장되는 값."""
    READY                = "ready"
    CHECKED_IN           = "checked_in"
    NEEDS_CONFIRMATION   = "needs_confirmation"   # 확인 필요 영역 감지, 학생 자료 제출 전
    PENDING_REVIEW       = "pending_review"
    APPROVED             = "approved"
    REJECTED             = "rejected"

    ALL: list[str] = [READY, CHECKED_IN, NEEDS_CONFIRMATION, PENDING_REVIEW, APPROVED, REJECTED]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.ALL


class IssueStatus:
    """
    issues.status 에 저장되는 값.

    설계 철학: "AI는 판정자가 아니라 확인 보조자입니다."
    AI는 전후 차이 후보를 감지하고, 학생은 확인 자료를 제출하며,
    관리자가 최종 판단합니다. 상태명은 색깔 중심이 아닌 의미 중심으로 정의합니다.
    """
    NEEDS_CONFIRMATION = "needs_confirmation"  # AI 감지 완료, 학생 확인 자료(근접 촬영+메모) 미제출
    EVIDENCE_SUBMITTED = "evidence_submitted"  # 학생이 근접 촬영과 메모를 제출한 상태
    CLEARED            = "cleared"             # 추가 확인이 필요 없다고 판단된 상태

    ALL: list[str] = [NEEDS_CONFIRMATION, EVIDENCE_SUBMITTED, CLEARED]

    # 제출 가능 상태 (needs_confirmation 이 하나라도 남아 있으면 제출 불가)
    RESOLVED: list[str] = [EVIDENCE_SUBMITTED, CLEARED]

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
