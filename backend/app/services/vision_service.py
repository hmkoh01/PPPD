"""
기숙사 퇴사 점검 — CV 비전 서비스.

align_and_detect() 한 함수가 전체 파이프라인을 담당합니다:
  1. ref 이미지 로드 (initial_image_path)
  2. final 이미지 디코딩 (bytes 또는 파일 경로)
  3. ORB 정합 (align_images)
  4. 정합 결과 저장 (aligned_final)
  5. SSIM 차이 검출 (detect_difference)
  6. 이슈 박스마다 크롭 이미지 저장
  7. VisionDetectionResult 반환

Gemini VLM 분석은 Phase 5 에서 별도 서비스로 구현합니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.constants import ImageType
from app.services.storage_service import (
    decode_image_bytes_to_bgr,
    load_image_bgr,
    save_ndarray,
)
from app.vision import DiffCandidate, ImageAlignmentError, align_images, detect_difference


# ────────────────────────────────────────────────────────────
# 커스텀 예외
# ────────────────────────────────────────────────────────────

class VisionServiceError(RuntimeError):
    """비전 서비스 전반의 기본 예외."""


class VisionAlignmentFailed(VisionServiceError):
    """이미지 정합(ORB 호모그래피)에 실패한 경우."""


class VisionDetectionFailed(VisionServiceError):
    """차이 검출 단계에서 예기치 않은 오류가 발생한 경우."""


# ────────────────────────────────────────────────────────────
# 결과 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class DetectedIssue:
    """
    감지된 단일 전후 차이 후보 영역.

    이 결과는 "손상 확정"이 아니라 "확인이 필요한 후보"입니다.
    학생의 근접 촬영 자료와 관리자의 최종 판단을 보조하기 위한 정보입니다.
    """
    x: int
    y: int
    width: int
    height: int
    crop_image_path: str | None = None  # DB 저장용 파일명 (상대 경로)
    score: float = 0.0                  # 후보 신뢰 점수 (0.0~1.0, 참고용)
    reason: str = "전후 이미지에서 국소적인 구조 차이가 감지됨"  # 감지 이유
    candidate_type: str = "small_damage"  # small_damage | large_object | recapture_recommended


@dataclass
class VisionDetectionResult:
    """align_and_detect() 반환 값."""
    aligned_final_image_path: str          # DB 저장용 파일명
    issues: list[DetectedIssue] = field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 내부 헬퍼
# ────────────────────────────────────────────────────────────

def clamp_box_to_image(
    x: int, y: int, w: int, h: int, image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """
    bounding box 좌표를 이미지 경계 안으로 클램프합니다.

    Parameters
    ----------
    x, y, w, h : int
        원본 bounding box (x 좌상단, y 좌상단, 너비, 높이).
    image_shape : tuple
        ``ndarray.shape`` — (height, width) 또는 (height, width, channels).

    Returns
    -------
    tuple[int, int, int, int]
        클램프된 (x, y, width, height). width/height 가 0 이하면 (0,0,0,0).
    """
    img_h, img_w = image_shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2 - x1, y2 - y1)


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """
    이미지에서 bounding box 영역을 잘라냅니다.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 이미지.
    box : tuple[int, int, int, int]
        (x, y, width, height) — clamp_box_to_image 를 거친 값이어야 합니다.

    Returns
    -------
    np.ndarray | None
        크롭된 BGR 이미지. box 가 유효하지 않으면 None.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    return image[y : y + h, x : x + w].copy()


def candidates_to_detected_issues(
    candidates: list[DiffCandidate],
    aligned_bgr: np.ndarray,
    inspection_id: int | None,
) -> list[DetectedIssue]:
    """
    detect_difference() 가 반환한 DiffCandidate 목록을 DetectedIssue 목록으로 변환합니다.
    각 후보마다 크롭 이미지를 저장하고 파일명을 DetectedIssue.crop_image_path 에 기록합니다.

    Parameters
    ----------
    candidates : list[DiffCandidate]
        detect_difference() 가 반환한 전후 차이 후보 목록.
    aligned_bgr : np.ndarray
        정합된 final 이미지 (크롭 소스).
    inspection_id : int | None
        DB inspection PK — 파일명 prefix 로 사용.

    Returns
    -------
    list[DetectedIssue]
        crop_image_path, score, reason 이 채워진 DetectedIssue 목록.
    """
    issues: list[DetectedIssue] = []
    for candidate in candidates:
        cx, cy, cw, ch = clamp_box_to_image(*candidate.box, aligned_bgr.shape)
        crop = crop_box(aligned_bgr, (cx, cy, cw, ch))
        crop_path: str | None = None
        if crop is not None:
            stored = save_ndarray(
                crop,
                ImageType.CROP,
                inspection_id=inspection_id,
            )
            crop_path = stored.filename
        issues.append(DetectedIssue(
            x=cx, y=cy, width=cw, height=ch,
            crop_image_path=crop_path,
            score=candidate.score,
            reason=candidate.reason,
            candidate_type=candidate.candidate_type,
        ))
    return issues


# ────────────────────────────────────────────────────────────
# 메인 파이프라인
# ────────────────────────────────────────────────────────────

def align_and_detect(
    initial_image_path: str,
    final_image_bytes: bytes | None = None,
    final_image_path: str | None = None,
    room_number: str | None = None,
    inspection_id: int | None = None,
) -> VisionDetectionResult:
    """
    기준(initial) 이미지와 최종(final) 이미지를 정합하고 차이 영역을 검출합니다.

    Parameters
    ----------
    initial_image_path : str
        DB 에 저장된 initial 이미지 파일명 (또는 절대경로).
        ref 이미지로 사용됩니다.
    final_image_bytes : bytes | None
        업로드된 final 이미지 raw bytes. ``final_image_path`` 와 둘 중 하나를 반드시 전달해야 합니다.
    final_image_path : str | None
        DB 에 저장된 final 이미지 파일명. ``final_image_bytes`` 가 None 일 때 사용됩니다.
    room_number : str | None
        이미지 파일명 prefix 용 호실 번호.
    inspection_id : int | None
        이미지 파일명 prefix 용 inspection PK.

    Returns
    -------
    VisionDetectionResult
        정합 이미지 파일명 + 검출된 이슈 목록.

    Raises
    ------
    ValueError
        final 이미지 소스가 하나도 제공되지 않은 경우.
    VisionAlignmentFailed
        ORB 정합 실패 (특징점 부족, 호모그래피 추정 실패 등).
    VisionDetectionFailed
        차이 검출 도중 예기치 않은 오류가 발생한 경우.
    """
    # ── 1. 이미지 로드 ──────────────────────────────────────
    try:
        ref_bgr = load_image_bgr(initial_image_path)
    except Exception as exc:
        raise VisionServiceError(
            f"initial 이미지를 로드할 수 없습니다: {initial_image_path!r} — {exc}"
        ) from exc

    if final_image_bytes is not None:
        try:
            curr_bgr = decode_image_bytes_to_bgr(final_image_bytes)
        except Exception as exc:
            raise VisionServiceError(
                f"final 이미지 bytes 디코딩 실패 — {exc}"
            ) from exc
    elif final_image_path is not None:
        try:
            curr_bgr = load_image_bgr(final_image_path)
        except Exception as exc:
            raise VisionServiceError(
                f"final 이미지를 로드할 수 없습니다: {final_image_path!r} — {exc}"
            ) from exc
    else:
        raise ValueError(
            "final_image_bytes 또는 final_image_path 중 하나를 반드시 전달해야 합니다."
        )

    # ── 2. ORB 정합 ─────────────────────────────────────────
    try:
        aligned_bgr = align_images(ref_bgr, curr_bgr)
    except ImageAlignmentError as exc:
        raise VisionAlignmentFailed(str(exc)) from exc
    except Exception as exc:
        raise VisionAlignmentFailed(
            f"이미지 정합 중 예기치 않은 오류가 발생했습니다: {exc}"
        ) from exc

    # ── 3. 정합 결과 저장 ───────────────────────────────────
    try:
        stored_aligned = save_ndarray(
            aligned_bgr,
            ImageType.ALIGNED_FINAL,
            room_number=room_number,
            inspection_id=inspection_id,
        )
    except Exception as exc:
        raise VisionServiceError(
            f"정합 이미지 저장 실패 — {exc}"
        ) from exc

    # ── 4. SSIM 전후 차이 후보 검출 ─────────────────────────
    try:
        candidates = detect_difference(ref_bgr, aligned_bgr)
    except Exception as exc:
        raise VisionDetectionFailed(
            f"차이 후보 검출 중 오류가 발생했습니다: {exc}"
        ) from exc

    # ── 5. 후보 크롭 이미지 저장 ────────────────────────────
    try:
        issues = candidates_to_detected_issues(candidates, aligned_bgr, inspection_id)
    except Exception as exc:
        raise VisionDetectionFailed(
            f"확인 필요 후보 크롭 이미지 저장 중 오류가 발생했습니다: {exc}"
        ) from exc

    return VisionDetectionResult(
        aligned_final_image_path=stored_aligned.filename,
        issues=issues,
    )
