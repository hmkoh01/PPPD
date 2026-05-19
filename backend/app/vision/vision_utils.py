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

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# 조정 가능한 상수
# ────────────────────────────────────────────────────────────

# 학생에게 표시할 최대 후보 수 (score 상위 N개)
MAX_CANDIDATES: int = 3

# score 기준 이하 후보는 표시하지 않음 (0.0~1.0)
# 손상 누락 시 낮추고, 오탐 과다 시 높임
DISPLAY_SCORE_THRESHOLD: float = 0.45

# Hard filter: 이보다 큰 aspect_ratio 는 즉시 제거
ASPECT_RATIO_HARD_LIMIT: float = 8.0

# Hard filter: 이보다 작은 fill_ratio 는 즉시 제거
FILL_RATIO_MIN: float = 0.15

# Hard filter: 이보다 작은 mean_diff 는 즉시 제거
MEAN_DIFF_MIN: float = 20.0

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
EDGE_MARGIN_RATIO: float = 0.05

# ORB 매칭·RANSAC 에 필요한 최소 좋은 매칭 개수
_MIN_GOOD_MATCHES = 15


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


# ────────────────────────────────────────────────────────────
# 차이 후보 검출
# ────────────────────────────────────────────────────────────

def detect_difference(
    ref_img: np.ndarray,
    aligned_curr_img: np.ndarray,
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
    gray_curr_raw = cv2.cvtColor(aligned_curr_img, cv2.COLOR_BGR2GRAY)

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
    _, valid_mask = cv2.threshold(gray_curr_raw, 1, 255, cv2.THRESH_BINARY)
    valid_mask    = cv2.erode(valid_mask, np.ones((40, 40), np.uint8), iterations=1)
    binary        = cv2.bitwise_and(binary, binary, mask=valid_mask)

    # ── Morphological open/close ─────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 후보 평가 루프 ────────────────────────────────────────
    raw_candidates: list[DiffCandidate] = []

    for c in contours:
        area = cv2.contourArea(c)

        # 면적 hard filter (최솟값)
        if area < MIN_AREA:
            continue
        # 90% 초과는 거의 이미지 전체 → 정합 실패로 판단하여 스킵
        if area > MAX_AREA:
            logger.debug(
                "후보 면적 초과(%.0f px², 한도 %.0f px²): 정합 실패 가능성으로 스킵",
                area, MAX_AREA,
            )
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

        x, y, bw, bh = cv2.boundingRect(c)
        bbox_area    = bw * bh

        # ── Hard filter ───────────────────────────────────────
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        fill_ratio   = area / max(bbox_area, 1)

        roi = diff_u8[y : y + bh, x : x + bw]
        mean_diff = float(np.mean(roi))

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
            boundary_score = min(edge_change / 30.0, 1.0)
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
            base_score = fill_score * 0.20 + diff_score * 0.35 + edge_score * 0.45

            # 조명/그림자 감점
            lighting_penalty = 0.0
            if edge_change < 10.0 and brightness_delta > 25.0:
                lighting_penalty = min(brightness_delta / 80.0, 1.0) * 0.40

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
    return above_threshold[:MAX_CANDIDATES]
