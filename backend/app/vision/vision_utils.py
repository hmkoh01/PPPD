"""
기숙사 퇴사 점검 — 이미지 정합 및 전후 차이 후보 검출 유틸리티.

설계 철학: AI는 판정자가 아니라 확인 보조자입니다.
  detect_difference() 는 전후 이미지에서 "확인이 필요한 후보 영역"을 감지합니다.
  결과는 손상 확정이 아니며, 학생의 근접 촬영 자료 및 관리자의 최종 판단을 보조합니다.

후보 선별 원칙:
  1차 감지는 넓게(SSIM 기반 고민감도) 유지하고, 후처리 필터로 학생에게 보여줄
  후보만 줄입니다. "민감도를 낮추는 것"이 아니라 "오탐을 걸러내는 것"이 목표입니다.

  주요 오탐 제거 기준:
    - 조명/그림자 변화: brightness_delta가 크고 edge_change가 작음
    - 정합 경계 아티팩트: 이미지 가장자리 근처 + 긴 형태(aspect_ratio 큼)
    - 희소 노이즈: fill_ratio 낮음
    - 미미한 차이: mean_diff 낮음
    - 너무 넓은 영역: 조명 변화 또는 정합 실패 가능성

  조정 가이드:
    손상을 놓침       → DISPLAY_SCORE_THRESHOLD 낮춤 (0.40)
    너무 많이 잡힘    → DISPLAY_SCORE_THRESHOLD 올림 (0.50)
    조명/그림자 잡힘  → brightness_delta 감점 강화
    가는 선이 잡힘    → ASPECT_RATIO_HARD_LIMIT 낮춤 (6)
    침구 주름이 잡힘  → FILL_RATIO_MIN 올림 (0.20)

변경 내역:
  - DiffCandidate 데이터클래스 추가 (box, score, reason, candidate_type)
  - detect_difference 반환 타입: list[tuple] → list[DiffCandidate]
  - 동적 면적 기준 (image_area 비율 기반)
  - edge_change / brightness_delta 특징 추가
  - 강화된 점수 계산 및 조명 변화 감점
  - DISPLAY_SCORE_THRESHOLD / MAX_CANDIDATES 제한
  - 이미지 가장자리 근접 감점
  - reason 문자열 특징별 자동 생성
  - candidate_type 분류: small_damage / large_object / recapture_recommended
  - large_object 전용 점수 계산 (objectness/diff_strength/boundary/locality)
  - large_object 정렬 우선순위 +0.10 부스트
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
import uuid

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.core.config import settings

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# 조정 가능한 상수
# ────────────────────────────────────────────────────────────

# 학생에게 표시할 최대 후보 수 (score 상위 N개)
MAX_CANDIDATES: int = 3

# score 기준 이하 후보는 표시하지 않음 (0.0~1.0)
# 손상 누락 시 낮추고, 오탐 과다 시 높임
DISPLAY_SCORE_THRESHOLD: float = 0.50

# Hard filter: 이보다 큰 aspect_ratio 는 즉시 제거
ASPECT_RATIO_HARD_LIMIT: float = 8.0

# Hard filter: 이보다 작은 fill_ratio 는 즉시 제거
FILL_RATIO_MIN: float = 0.18

# Hard filter: 이보다 작은 mean_diff 는 즉시 제거
MEAN_DIFF_MIN: float = 24.0
GRADIENT_RESIDUAL_MIN: float = 14.0
CHROMA_DELTA_MIN: float = 8.0

# 이미지 면적 대비 동적 최소/최대 면적 비율 (small_damage 기준)
IMAGE_AREA_MIN_RATIO: float = 0.002   # image_area * 0.002 (최소 1500 px²)
IMAGE_AREA_MAX_RATIO: float = 0.20    # 20% 미만 = small_damage 영역

# 큰 객체/방치물 감지 범위
LARGE_OBJECT_MIN_AREA_RATIO: float = 0.01   # 1% 이상 → large_object 후보로 평가
LARGE_OBJECT_MAX_AREA_RATIO: float = 0.60   # 60% 초과 → recapture_recommended

# 감지 프로파일: "auto" 모드에서 면적 비율로 candidate_type 자동 결정
# "auto" | "small_damage" | "large_object"
DETECTION_PROFILE: str = "auto"

# 이미지 가장자리 margin 비율 (이 범위 안에 있으면 감점)
EDGE_MARGIN_RATIO: float = 0.10
LARGE_OBJECT_MIN_EDGE_CHANGE: float = 16.0
LARGE_OBJECT_MIN_GRADIENT_RESIDUAL: float = 28.0
SMALL_DAMAGE_MIN_GRADIENT_RESIDUAL: float = 68.0

# ORB 매칭·RANSAC 에 필요한 최소 좋은 매칭 개수
_MIN_GOOD_MATCHES = 15

ALIGNMENT_LOCKED_THRESHOLD: float = 0.76
ALIGNMENT_GOOD_THRESHOLD: float = 0.78
ALIGNMENT_ALMOST_THRESHOLD: float = 0.64
ALIGNMENT_TARGET_MATCHES: int = 65
ALIGNMENT_LOCKED_MIN_INLIER_RATIO: float = 0.60
ALIGNMENT_LOCKED_MAX_BORDER_RATIO: float = 0.145
ALIGNMENT_LOCKED_MIN_SHAPE_SCORE: float = 0.66
ALIGNMENT_LOCKED_MIN_SIMILARITY_SCORE: float = 0.60
ALIGNMENT_LOCKED_MIN_SSIM_RAW: float = 0.64
ALIGNMENT_LOCKED_MIN_RESIDUAL_SCORE: float = 0.49
ALIGNMENT_MAX_BORDER_RATIO: float = 0.14
ALIGNMENT_GRADIENT_RESIDUAL_MAX: float = 47.0
ALIGNMENT_EDGE_RESIDUAL_MAX: float = 71.0
DIFF_DEBUG_IMAGES_ENABLED: bool = True


# ────────────────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class DiffCandidate:
    """
    detect_difference() 가 감지한 단일 전후 차이 후보 영역.

    이 결과는 "손상 확정"이 아니라 "확인이 필요한 후보"입니다.
    학생의 근접 촬영 자료와 관리자의 최종 판단을 통해 결론이 내려집니다.

    Attributes
    ----------
    box : tuple[int, int, int, int]
        (x, y, width, height) 픽셀 좌표.
    score : float
        0.0~1.0 범위의 후보 신뢰 점수.
    reason : str
        감지 이유 및 특징 설명 (관리자 참고용 한국어).
    candidate_type : str
        후보 유형.
        - "small_damage"          : 작은 손상/오염 후보 (면적 < 1%)
        - "large_object"          : 큰 객체/방치물 후보 (면적 1~60%)
        - "recapture_recommended" : 이미지 전체 변화 감지 — 재촬영 권장 (면적 > 60%)
    """
    box: tuple[int, int, int, int]
    score: float
    reason: str
    candidate_type: str = "small_damage"


@dataclass
class AlignmentCheckResult:
    """Lightweight alignment quality result for pre-upload capture validation."""
    ok: bool
    score: float
    status: str
    message: str
    hints: list[str]
    good_matches: int | None = None
    inlier_ratio: float | None = None
    ssim_score: float | None = None
    residual_score: float | None = None
    ssim_ready: bool = False


# ────────────────────────────────────────────────────────────
# 예외
# ────────────────────────────────────────────────────────────

class ImageAlignmentError(RuntimeError):
    """
    두 이미지 간 특징점 매칭 또는 호모그래피 추정이 실패하여
    기하 정합(alignment)을 수행할 수 없을 때 발생하는 예외입니다.
    """
    pass


# ────────────────────────────────────────────────────────────
# 정합
# ────────────────────────────────────────────────────────────

def align_images(ref_img: np.ndarray, curr_img: np.ndarray) -> np.ndarray:
    """
    ORB 특징점과 BFMatcher 로 대응점을 찾고, RANSAC 으로 호모그래피를 추정한 뒤
    ``curr_img`` 를 ``ref_img`` 의 좌표계·시야에 맞게 투시 변환합니다.

    Parameters
    ----------
    ref_img : np.ndarray
        기준 이미지(BGR, uint8). 출력 해상도 및 좌표계의 기준이 됩니다.
    curr_img : np.ndarray
        현재 촬영 이미지(BGR, uint8). 이 이미지가 정합되어 반환됩니다.

    Returns
    -------
    np.ndarray
        ``ref_img`` 와 동일한 크기로 워핑된 정합 결과(BGR, uint8).

    Raises
    ------
    ImageAlignmentError
        특징점 부족, 매칭 실패, RANSAC 호모그래피 추정 실패 등.
    ValueError
        올바른 3채널 BGR 이미지가 아니거나 비어 있는 경우.
    """
    if ref_img is None or curr_img is None:
        raise ValueError("ref_img 와 curr_img 는 None 일 수 없습니다.")
    if ref_img.size == 0 or curr_img.size == 0:
        raise ValueError("입력 이미지가 비어 있습니다.")
    if ref_img.ndim != 3 or ref_img.shape[2] != 3:
        raise ValueError("ref_img 는 HxWx3 BGR 이미지여야 합니다.")
    if curr_img.ndim != 3 or curr_img.shape[2] != 3:
        raise ValueError("curr_img 는 HxWx3 BGR 이미지여야 합니다.")

    ref_gray  = cv2.cvtColor(ref_img,  cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_gray,  None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ImageAlignmentError(
            "한쪽 또는 양쪽 이미지에서 ORB 특징점을 충분히 찾지 못했습니다. "
            "조명·초점·동일 장소 촬영 여부를 확인해 주세요."
        )

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)

    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < _MIN_GOOD_MATCHES:
        raise ImageAlignmentError(
            f"신뢰할 만한 특징점 매칭이 부족합니다(좋은 매칭 {len(good)}개, "
            f"필요 {_MIN_GOOD_MATCHES}개 이상). 같은 장소에서 비슷한 각도로 다시 촬영해 보세요."
        )

    src_pts = np.float32([kp2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good]).reshape(-1, 1, 2)

    H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ImageAlignmentError(
            "RANSAC 으로 호모그래피를 추정하지 못했습니다. 장면 변화가 크거나 "
            "반복 패턴이 많을 수 있습니다."
        )

    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(curr_img, H, (w, h))
    return aligned


def _alignment_message(status: str) -> str:
    if status == "locked":
        return "정렬 완료. 자동으로 촬영할게요."
    if status == "good":
        return "거의 맞았어요. 그대로 유지해 주세요."
    if status == "almost":
        return "조금만 더 맞춰주세요."
    return "기준 사진과 아직 어긋나 있어요."


def _homography_shape_penalty(H: np.ndarray) -> float:
    corners = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    widths = [
        float(np.linalg.norm(warped[1] - warped[0])),
        float(np.linalg.norm(warped[2] - warped[3])),
    ]
    heights = [
        float(np.linalg.norm(warped[3] - warped[0])),
        float(np.linalg.norm(warped[2] - warped[1])),
    ]
    max_width = max(widths)
    min_width = max(min(widths), 1e-6)
    max_height = max(heights)
    min_height = max(min(heights), 1e-6)
    skew = max(max_width / min_width, max_height / min_height)
    return min(max(skew - 1.0, 0.0) / 1.2, 1.0)


def _normalize_brightness_color(reference_bgr: np.ndarray, candidate_bgr: np.ndarray) -> np.ndarray:
    """Match candidate LAB channel statistics to the reference before SSIM."""
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cand_lab = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    normalized = cand_lab.copy()

    for channel in range(3):
        ref_channel = ref_lab[:, :, channel]
        cand_channel = cand_lab[:, :, channel]
        ref_mean, ref_std = cv2.meanStdDev(ref_channel)
        cand_mean, cand_std = cv2.meanStdDev(cand_channel)
        ref_mean_f = float(ref_mean[0][0])
        ref_std_f = max(float(ref_std[0][0]), 1.0)
        cand_mean_f = float(cand_mean[0][0])
        cand_std_f = max(float(cand_std[0][0]), 1.0)
        normalized[:, :, channel] = ((cand_channel - cand_mean_f) * (ref_std_f / cand_std_f)) + ref_mean_f

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def _alignment_residual_scores(ref_gray: np.ndarray, aligned_gray: np.ndarray) -> tuple[float, float, float]:
    ref_grad = _gradient_magnitude(ref_gray)
    aligned_grad = _gradient_magnitude(aligned_gray)
    gradient_residual = float(np.mean(cv2.absdiff(ref_grad, aligned_grad)))
    gradient_score = 1.0 - min(gradient_residual / ALIGNMENT_GRADIENT_RESIDUAL_MAX, 1.0)

    ref_edges = cv2.Canny(ref_gray, 50, 150)
    aligned_edges = cv2.Canny(aligned_gray, 50, 150)
    edge_residual = float(np.mean(cv2.absdiff(ref_edges, aligned_edges)))
    edge_score = 1.0 - min(edge_residual / ALIGNMENT_EDGE_RESIDUAL_MAX, 1.0)

    residual_score = float(np.clip((gradient_score * 0.6) + (edge_score * 0.4), 0.0, 1.0))
    return residual_score, gradient_residual, edge_residual


def _save_diff_debug_image(
    debug_prefix: str | None,
    case: str,
    ref_img: np.ndarray,
    curr_img: np.ndarray,
    diff_u8: np.ndarray,
    binary: np.ndarray,
    box: tuple[int, int, int, int] | None = None,
    note: str = "",
) -> None:
    if not DIFF_DEBUG_IMAGES_ENABLED or not debug_prefix:
        return
    try:
        storage_dir = settings.image_storage_path
        storage_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = storage_dir / "vision_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        ref_vis = ref_img.copy()
        curr_vis = curr_img.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(ref_vis, (x, y), (x + w, y + h), (0, 180, 255), 3)
            cv2.rectangle(curr_vis, (x, y), (x + w, y + h), (0, 180, 255), 3)

        diff_vis = cv2.cvtColor(diff_u8, cv2.COLOR_GRAY2BGR)
        binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        panels = [ref_vis, curr_vis, diff_vis, binary_vis]
        target_h = 260
        resized: list[np.ndarray] = []
        for panel in panels:
            scale = target_h / max(panel.shape[0], 1)
            resized.append(cv2.resize(panel, (max(1, int(panel.shape[1] * scale)), target_h)))
        sheet = cv2.hconcat(resized)
        label = f"{case} {note}"[:160]
        cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(sheet, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        safe_case = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case)[:60]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{debug_prefix}_{safe_case}_{ts}_{uuid.uuid4().hex[:6]}.jpg"
        encoded, data = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if encoded:
            with open(debug_dir / filename, "wb") as f:
                f.write(data.tobytes())
    except Exception:
        logger.debug("failed to save diff debug image", exc_info=True)


def check_alignment_quality(ref_img: np.ndarray, curr_img: np.ndarray) -> AlignmentCheckResult:
    """
    Estimate whether the current capture is close enough to the reference.

    This function does not save images and does not run difference detection.
    It reuses ORB matching and homography estimation, then combines match count,
    RANSAC inlier ratio, transform distortion, post-warp similarity, and black
    border ratio into a normalized 0.0-1.0 score.
    """
    if ref_img is None or curr_img is None:
        raise ValueError("ref_img and curr_img must not be None.")
    if ref_img.size == 0 or curr_img.size == 0:
        raise ValueError("input images must not be empty.")
    if ref_img.ndim != 3 or ref_img.shape[2] != 3:
        raise ValueError("ref_img must be an HxWx3 BGR image.")
    if curr_img.ndim != 3 or curr_img.shape[2] != 3:
        raise ValueError("curr_img must be an HxWx3 BGR image.")

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return AlignmentCheckResult(
            ok=False,
            score=0.0,
            status="poor",
            message=_alignment_message("poor"),
            hints=["move_phone"],
            good_matches=0,
            inlier_ratio=0.0,
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)

    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good_matches = len(good)
    if good_matches < 4:
        return AlignmentCheckResult(
            ok=False,
            score=0.0,
            status="poor",
            message=_alignment_message("poor"),
            hints=["move_phone"],
            good_matches=good_matches,
            inlier_ratio=0.0,
        )

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        match_score = min(good_matches / ALIGNMENT_TARGET_MATCHES, 1.0)
        score = round(match_score * 0.35, 3)
        return AlignmentCheckResult(
            ok=False,
            score=score,
            status="poor",
            message=_alignment_message("poor"),
            hints=["reduce_tilt"],
            good_matches=good_matches,
            inlier_ratio=0.0,
        )

    inlier_ratio = float(mask.ravel().sum()) / max(good_matches, 1)
    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(curr_img, H, (w, h))
    normalized_aligned = _normalize_brightness_color(ref_img, aligned)

    aligned_gray = cv2.cvtColor(normalized_aligned, cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (15, 15), 0)
    aligned_blur = cv2.GaussianBlur(aligned_gray, (15, 15), 0)
    similarity, _ = ssim(ref_blur, aligned_blur, win_size=21, data_range=255, full=True)
    similarity_score = float(np.clip((similarity - 0.35) / 0.45, 0.0, 1.0))
    residual_score, gradient_residual, edge_residual = _alignment_residual_scores(ref_gray, aligned_gray)

    source_mask = np.full(curr_gray.shape, 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(source_mask, H, (w, h))
    border_ratio = 1.0 - (float(np.count_nonzero(warped_mask)) / float(warped_mask.size))
    border_score = 1.0 - min(border_ratio / ALIGNMENT_MAX_BORDER_RATIO, 1.0)

    match_score = min(good_matches / ALIGNMENT_TARGET_MATCHES, 1.0)
    inlier_score = min(inlier_ratio / 0.75, 1.0)
    shape_score = 1.0 - _homography_shape_penalty(H)

    score = (
        match_score * 0.25
        + inlier_score * 0.25
        + similarity_score * 0.22
        + residual_score * 0.08
        + border_score * 0.12
        + shape_score * 0.08
    )
    score = round(float(np.clip(score, 0.0, 1.0)), 3)

    ssim_ready = (
        similarity >= ALIGNMENT_LOCKED_MIN_SSIM_RAW
        and residual_score >= ALIGNMENT_LOCKED_MIN_RESIDUAL_SCORE
        and gradient_residual <= ALIGNMENT_GRADIENT_RESIDUAL_MAX
        and edge_residual <= ALIGNMENT_EDGE_RESIDUAL_MAX
    )
    locked = (
        score >= ALIGNMENT_LOCKED_THRESHOLD
        and inlier_ratio >= ALIGNMENT_LOCKED_MIN_INLIER_RATIO
        and border_ratio <= ALIGNMENT_LOCKED_MAX_BORDER_RATIO
        and shape_score >= ALIGNMENT_LOCKED_MIN_SHAPE_SCORE
        and similarity_score >= ALIGNMENT_LOCKED_MIN_SIMILARITY_SCORE
        and ssim_ready
    )

    if locked:
        status = "locked"
    elif score >= ALIGNMENT_GOOD_THRESHOLD:
        status = "good"
    elif score >= ALIGNMENT_ALMOST_THRESHOLD:
        status = "almost"
    else:
        status = "poor"

    hints: list[str] = []
    if status != "locked":
        if border_ratio > ALIGNMENT_MAX_BORDER_RATIO:
            hints.append("move_closer")
        if shape_score < ALIGNMENT_LOCKED_MIN_SHAPE_SCORE:
            hints.append("reduce_tilt")
        if not ssim_ready:
            hints.append("align_edges")
        if not hints:
            hints.append("move_phone")

    return AlignmentCheckResult(
        ok=locked,
        score=score,
        status=status,
        message=_alignment_message(status),
        hints=hints,
        good_matches=good_matches,
        inlier_ratio=round(inlier_ratio, 3),
        ssim_score=round(float(similarity), 3),
        residual_score=round(residual_score, 3),
        ssim_ready=ssim_ready,
    )


# ────────────────────────────────────────────────────────────
# 차이 후보 검출
# ────────────────────────────────────────────────────────────

def detect_difference(
    ref_img: np.ndarray,
    aligned_curr_img: np.ndarray,
    debug_prefix: str | None = None,
) -> list[DiffCandidate]:
    """
    정합된 두 이미지 간의 SSIM 전후 차이 후보를 검출합니다.

    1차 감지(SSIM)는 넓게 유지하고, 이후 다단계 필터와 점수 계산으로
    조명·그림자·침구 주름·정합 오차 등 오탐을 제거합니다.
    최종적으로 score 상위 MAX_CANDIDATES 개만 반환합니다.

    Parameters
    ----------
    ref_img : np.ndarray
        기준 이미지(BGR, uint8).
    aligned_curr_img : np.ndarray
        ref_img 와 동일한 해상도로 정합된 현재 이미지(BGR, uint8).

    Returns
    -------
    list[DiffCandidate]
        score 기준 내림차순 정렬된 전후 차이 후보 목록 (최대 MAX_CANDIDATES 개).

    Raises
    ------
    ValueError
        입력이 None 이거나 두 이미지의 크기가 다를 때.
    """
    if ref_img is None or aligned_curr_img is None:
        raise ValueError("ref_img 와 aligned_curr_img 는 None 일 수 없습니다.")
    if ref_img.shape[:2] != aligned_curr_img.shape[:2]:
        raise ValueError("이미지 크기가 일치해야 합니다.")

    img_h, img_w = ref_img.shape[:2]
    image_area   = img_h * img_w

    # ── 동적 면적 기준 ────────────────────────────────────────
    MIN_AREA = max(1500, image_area * IMAGE_AREA_MIN_RATIO)
    # 최대 면적: 90% 초과는 정합 실패로 판단하여 스킵
    # 60~90% 범위는 recapture_recommended 로 분류
    MAX_AREA = image_area * 0.90

    # 이미지 가장자리 margin
    margin_x = img_w * EDGE_MARGIN_RATIO
    margin_y = img_h * EDGE_MARGIN_RATIO

    # ── 그레이스케일 변환 ─────────────────────────────────────
    # raw: edge/brightness 계산용 (비블러)
    # blurred: SSIM 계산용
    gray_ref_raw  = cv2.cvtColor(ref_img,          cv2.COLOR_BGR2GRAY)
    normalized_curr_img = _normalize_brightness_color(ref_img, aligned_curr_img)
    gray_curr_raw = cv2.cvtColor(normalized_curr_img, cv2.COLOR_BGR2GRAY)

    gray_ref_blur  = cv2.GaussianBlur(gray_ref_raw,  (15, 15), 0)
    gray_curr_blur = cv2.GaussianBlur(gray_curr_raw, (15, 15), 0)

    # ── SSIM 차이 맵 ──────────────────────────────────────────
    _, ssim_map = ssim(gray_ref_blur, gray_curr_blur, win_size=21, data_range=255, full=True)

    diff    = np.clip(1.0 - ssim_map, 0.0, 1.0)
    diff_u8 = (diff * 255).astype(np.uint8)

    # ── Otsu threshold (최소 100 보장) ───────────────────────
    t_otsu, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_eff     = max(float(t_otsu), 100.0)
    _, binary = cv2.threshold(diff_u8, t_eff, 255, cv2.THRESH_BINARY)

    # ── valid_mask: warpPerspective 검은 테두리 제거 ─────────
    aligned_raw_gray = cv2.cvtColor(aligned_curr_img, cv2.COLOR_BGR2GRAY)
    _, valid_mask = cv2.threshold(aligned_raw_gray, 1, 255, cv2.THRESH_BINARY)
    valid_mask    = cv2.erode(valid_mask, np.ones((40, 40), np.uint8), iterations=1)
    binary        = cv2.bitwise_and(binary, binary, mask=valid_mask)

    # ── Morphological open/close ─────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 후보 평가 루프 ────────────────────────────────────────
    raw_candidates: list[DiffCandidate] = []

    for index, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        box = (x, y, bw, bh)

        # 면적 hard filter (최솟값)
        if area < MIN_AREA:
            _save_diff_debug_image(debug_prefix, "filtered_area_too_small", ref_img, aligned_curr_img, diff_u8, binary, box, f"idx={index} area={area:.0f}")
            continue
        # 90% 초과는 거의 이미지 전체 → 정합 실패로 판단하여 스킵
        if area > MAX_AREA:
            logger.debug(
                "후보 면적 초과(%.0f px², 한도 %.0f px²): 정합 실패 가능성으로 스킵",
                area, MAX_AREA,
            )
            _save_diff_debug_image(debug_prefix, "filtered_area_too_large", ref_img, aligned_curr_img, diff_u8, binary, box, f"idx={index} area={area:.0f}")
            continue

        area_ratio = area / image_area

        # ── candidate_type 결정 (DETECTION_PROFILE="auto") ───
        if area_ratio > LARGE_OBJECT_MAX_AREA_RATIO:
            # 60~90% 범위: 이미지 전체 변화 — 재촬영 권장
            candidate_type = "recapture_recommended"
        elif area_ratio >= LARGE_OBJECT_MIN_AREA_RATIO:
            # 1~60% 범위: 큰 객체/방치물 후보
            candidate_type = "large_object"
        else:
            # 1% 미만: 소규모 손상/오염 후보
            candidate_type = "small_damage"

        bbox_area    = bw * bh

        # ── Hard filter ───────────────────────────────────────
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        fill_ratio   = area / max(bbox_area, 1)

        roi = diff_u8[y : y + bh, x : x + bw]
        mean_diff = float(np.mean(roi))

        if aspect_ratio > ASPECT_RATIO_HARD_LIMIT:
            _save_diff_debug_image(debug_prefix, "filtered_aspect_ratio", ref_img, aligned_curr_img, diff_u8, binary, box, f"idx={index} aspect={aspect_ratio:.2f}")
            continue
        if fill_ratio < FILL_RATIO_MIN:
            _save_diff_debug_image(debug_prefix, "filtered_sparse_fill", ref_img, aligned_curr_img, diff_u8, binary, box, f"idx={index} fill={fill_ratio:.2f}")
            continue
        if mean_diff < MEAN_DIFF_MIN:
            _save_diff_debug_image(debug_prefix, "filtered_weak_diff", ref_img, aligned_curr_img, diff_u8, binary, box, f"idx={index} mean={mean_diff:.1f}")
            continue

        if aspect_ratio > ASPECT_RATIO_HARD_LIMIT:
            continue  # 얇고 긴 형태 — 정합 경계 아티팩트
        if fill_ratio < FILL_RATIO_MIN:
            continue  # 희소 픽셀 — 노이즈
        if mean_diff < MEAN_DIFF_MIN:
            continue  # 미미한 차이 — 무시

        # ── 특징 계산 ─────────────────────────────────────────
        y1, y2 = max(0, y), min(img_h, y + bh)
        x1, x2 = max(0, x), min(img_w, x + bw)

        ref_crop  = gray_ref_raw[y1:y2, x1:x2]
        curr_crop = gray_curr_raw[y1:y2, x1:x2]

        if ref_crop.size == 0 or curr_crop.size == 0:
            continue

        # 평균 밝기 차이 (조명/전역 변화 지표)
        brightness_delta = abs(float(np.mean(ref_crop)) - float(np.mean(curr_crop)))

        # Canny edge 변화량 (구조적 경계 변화 지표)
        ref_edges   = cv2.Canny(ref_crop,  50, 150)
        curr_edges  = cv2.Canny(curr_crop, 50, 150)
        edge_change = float(np.mean(cv2.absdiff(ref_edges, curr_edges)))
        gradient_residual = float(np.mean(cv2.absdiff(_gradient_magnitude(ref_crop), _gradient_magnitude(curr_crop))))

        ref_lab_crop = cv2.cvtColor(ref_img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
        curr_lab_crop = cv2.cvtColor(normalized_curr_img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
        chroma_delta = float(np.mean(cv2.absdiff(ref_lab_crop[:, :, 1:], curr_lab_crop[:, :, 1:])))

        if gradient_residual < GRADIENT_RESIDUAL_MIN and chroma_delta < CHROMA_DELTA_MIN:
            _save_diff_debug_image(
                debug_prefix,
                "filtered_low_structure_residual",
                ref_img,
                aligned_curr_img,
                diff_u8,
                binary,
                box,
                f"idx={index} grad={gradient_residual:.1f} chroma={chroma_delta:.1f}",
            )
            continue

        if candidate_type == "large_object" and (
            edge_change < LARGE_OBJECT_MIN_EDGE_CHANGE
            or gradient_residual < LARGE_OBJECT_MIN_GRADIENT_RESIDUAL
        ):
            _save_diff_debug_image(
                debug_prefix,
                "filtered_large_object_weak_structure",
                ref_img,
                aligned_curr_img,
                diff_u8,
                binary,
                box,
                f"idx={index} edge={edge_change:.1f} grad={gradient_residual:.1f}",
            )
            continue

        if candidate_type == "small_damage" and gradient_residual < SMALL_DAMAGE_MIN_GRADIENT_RESIDUAL:
            _save_diff_debug_image(
                debug_prefix,
                "filtered_small_damage_weak_gradient",
                ref_img,
                aligned_curr_img,
                diff_u8,
                binary,
                box,
                f"idx={index} grad={gradient_residual:.1f}",
            )
            continue

        # 이미지 가장자리 근접 여부
        is_near_edge = (
            x < margin_x or y < margin_y
            or x + bw > img_w - margin_x
            or y + bh > img_h - margin_y
        )

        # aspect_ratio 감점 (두 경로 공통)
        aspect_penalty = 0.0
        if aspect_ratio > 4.0:
            aspect_penalty = min((aspect_ratio - 4.0) / 4.0, 1.0) * 0.25

        # 이미지 가장자리 감점
        edge_proximity_penalty = 0.15 if is_near_edge else 0.0

        # ── Score 계산: candidate_type 별 분기 ───────────────

        if candidate_type == "large_object":
            # 큰 객체/방치물 전용 점수
            # objectness(30%): 영역 밀도
            objectness_score = min(max(fill_ratio - FILL_RATIO_MIN, 0.0) / (1.0 - FILL_RATIO_MIN), 1.0)
            # diff_strength(30%): 전후 차이 강도
            diff_strength_score = min(max(mean_diff - MEAN_DIFF_MIN, 0.0) / (255.0 - MEAN_DIFF_MIN), 1.0)
            # boundary_score(25%): 경계 edge 변화 (객체 윤곽)
            boundary_score = min(max(edge_change, gradient_residual) / 34.0, 1.0)
            # locality_score(15%): 전역 밝기 변화가 낮을수록 국소 변화로 신뢰
            locality_score = max(0.0, 1.0 - brightness_delta / 80.0)

            base_score = (
                objectness_score   * 0.30
                + diff_strength_score * 0.30
                + boundary_score      * 0.25
                + locality_score      * 0.15
            )
            score = round(
                max(0.0, min(1.0, base_score - aspect_penalty - edge_proximity_penalty)),
                3,
            )

            if boundary_score > 0.5:
                reason = (
                    f"큰 객체 또는 방치물이 감지됨 (면적 {area_ratio:.0%})"
                    " — 경계 edge 변화가 뚜렷함"
                )
            else:
                reason = (
                    f"큰 면적 전후 차이가 감지됨 (면적 {area_ratio:.0%})"
                    " — 방치물 또는 가구 이동 가능성"
                )

        elif candidate_type == "recapture_recommended":
            # 이미지 전체 수준 변화 — 재촬영 권장
            diff_score = min(max(mean_diff - MEAN_DIFF_MIN, 0.0) / (255.0 - MEAN_DIFF_MIN), 1.0)
            score = round(min(diff_score * 0.6 + 0.20, 0.60), 3)
            reason = (
                f"이미지 전체의 큰 변화가 감지됨 (면적 {area_ratio:.0%})"
                " — 재촬영을 권장합니다"
            )

        else:
            # small_damage: 기존 점수 계산
            fill_score = min(max(fill_ratio - FILL_RATIO_MIN, 0.0) / (1.0 - FILL_RATIO_MIN), 1.0)
            diff_score = min(max(mean_diff - MEAN_DIFF_MIN, 0.0) / (255.0 - MEAN_DIFF_MIN), 1.0)
            edge_score = min(edge_change / 30.0, 1.0)
            gradient_score = min(gradient_residual / 34.0, 1.0)
            chroma_score = min(chroma_delta / 30.0, 1.0)
            base_score = (
                fill_score * 0.16
                + diff_score * 0.28
                + edge_score * 0.28
                + gradient_score * 0.20
                + chroma_score * 0.08
            )

            # 조명/그림자 감점
            lighting_penalty = 0.0
            if max(edge_change, gradient_residual) < 16.0 and brightness_delta > 20.0:
                lighting_penalty = min(brightness_delta / 70.0, 1.0) * 0.48

            score = round(
                max(0.0, min(1.0, base_score - lighting_penalty - aspect_penalty - edge_proximity_penalty)),
                3,
            )

            if lighting_penalty > 0.15:
                reason = (
                    f"밝기 변화가 크고(Δ{brightness_delta:.0f}) edge 변화가 작아"
                    " 조명 또는 그림자 영향 가능성이 있음"
                )
            elif is_near_edge and aspect_ratio > 4.0:
                reason = "이미지 가장자리 근처 후보로 정합 오차 가능성이 있음"
            elif edge_score > 0.5:
                reason = "국소적인 구조 차이와 edge 변화가 함께 감지됨"
            elif diff_score > 0.5:
                reason = "전후 이미지에서 국소적인 구조 차이가 감지됨"
            else:
                reason = "미세한 전후 차이가 감지됨 — 추가 확인 필요"

        raw_candidates.append(
            DiffCandidate(box=(x, y, bw, bh), score=score, reason=reason, candidate_type=candidate_type)
        )

    # ── 디버그 로그 ───────────────────────────────────────────
    above_threshold = [c for c in raw_candidates if c.score >= DISPLAY_SCORE_THRESHOLD]
    for c in raw_candidates:
        if c.score < DISPLAY_SCORE_THRESHOLD:
            _save_diff_debug_image(
                debug_prefix,
                "filtered_low_score",
                ref_img,
                aligned_curr_img,
                diff_u8,
                binary,
                c.box,
                f"score={c.score:.3f} type={c.candidate_type}",
            )
    logger.debug(
        "전후 차이 후보 총 %d건 | score threshold(%.2f) 이상: %d건 | 최종 표시: 최대 %d건",
        len(raw_candidates),
        DISPLAY_SCORE_THRESHOLD,
        len(above_threshold),
        MAX_CANDIDATES,
    )
    if len(raw_candidates) > MAX_CANDIDATES * 3:
        logger.warning(
            "후보가 매우 많음(%d건): 사진 구도 또는 조명 차이가 커서 재촬영이 필요할 수 있습니다.",
            len(raw_candidates),
        )

    # ── Score threshold 적용 → 상위 MAX_CANDIDATES 개 반환 ──
    # large_object 는 정렬 시 +0.10 우선순위 부스트 (표시 점수 자체는 변경하지 않음)
    above_threshold.sort(
        key=lambda c: c.score + (0.10 if c.candidate_type == "large_object" else 0.0),
        reverse=True,
    )
    final_candidates = above_threshold[:MAX_CANDIDATES]
    for c in final_candidates:
        _save_diff_debug_image(
            debug_prefix,
            "selected_candidate",
            ref_img,
            aligned_curr_img,
            diff_u8,
            binary,
            c.box,
            f"score={c.score:.3f} type={c.candidate_type}",
        )
    return final_candidates
