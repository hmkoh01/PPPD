"""
기숙사 퇴사 점검 PoC용 이미지 정합 및 차이 검출 유틸리티.

OpenCV(ORB, Homography)와 scikit-image(SSIM)를 사용합니다.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ORB 매칭·RANSAC에 필요한 최소 좋은 매칭 개수 (너무 적으면 호모그래피가 불안정합니다)
_MIN_GOOD_MATCHES = 15
# 노이즈로 간주하여 무시할 최소 윤곽선 면적(픽셀)
_MIN_CONTOUR_AREA = 500


class ImageAlignmentError(RuntimeError):
    """
    두 이미지 간 특징점 매칭 또는 호모그래피 추정이 실패하여
    기하 정합(alignment)을 수행할 수 없을 때 발생하는 예외입니다.
    """

    pass


def align_images(ref_img: np.ndarray, curr_img: np.ndarray) -> np.ndarray:
    """
    ORB 특징점과 BFMatcher로 대응점을 찾고, RANSAC으로 호모그래피를 추정한 뒤
    ``curr_img``를 ``ref_img``의 좌표계·시야에 맞게 투시 변환합니다.

    Parameters
    ----------
    ref_img : np.ndarray
        기준 이미지(BGR, ``uint8``). 출력 해상도 및 좌표계의 기준이 됩니다.
    curr_img : np.ndarray
        현재 촬영 이미지(BGR, ``uint8``). 이 이미지가 정합되어 반환됩니다.

    Returns
    -------
    np.ndarray
        ``ref_img``와 동일한 크기(width, height)로 워핑된 정합 결과(BGR, ``uint8``).

    Raises
    ------
    ImageAlignmentError
        특징점 부족, 매칭 실패, RANSAC 호모그래피 추정 실패 등으로 정합할 수 없는 경우.
    ValueError
        입력이 올바른 3채널 BGR 이미지가 아니거나 비어 있는 경우.

    Notes
    -----
    - Lowe 비율 테스트로 불안정한 매칭을 줄입니다.
    - ``cv2.warpPerspective``의 출력 크기는 ``ref_img.shape[1]``, ``ref_img.shape[0]``입니다.
    """
    if ref_img is None or curr_img is None:
        raise ValueError("ref_img와 curr_img는 None일 수 없습니다.")
    if ref_img.size == 0 or curr_img.size == 0:
        raise ValueError("입력 이미지가 비어 있습니다.")
    if ref_img.ndim != 3 or ref_img.shape[2] != 3:
        raise ValueError("ref_img는 HxWx3 BGR 이미지여야 합니다.")
    if curr_img.ndim != 3 or curr_img.shape[2] != 3:
        raise ValueError("curr_img는 HxWx3 BGR 이미지여야 합니다.")

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ImageAlignmentError(
            "한쪽 또는 양쪽 이미지에서 ORB 특징점을 충분히 찾지 못했습니다. "
            "조명·초점·동일 장소 촬영 여부를 확인해 주세요."
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

    if len(good) < _MIN_GOOD_MATCHES:
        raise ImageAlignmentError(
            f"신뢰할 만한 특징점 매칭이 부족합니다(좋은 매칭 {len(good)}개, "
            f"필요 {_MIN_GOOD_MATCHES}개 이상). 같은 장소에서 비슷한 각도로 다시 촬영해 보세요."
        )

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ImageAlignmentError(
            "RANSAC으로 호모그래피를 추정하지 못했습니다. 장면 변화가 크거나 "
            "반복 패턴이 많을 수 있습니다."
        )

    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(curr_img, H, (w, h))
    return aligned


def detect_difference(
    ref_img: np.ndarray, aligned_curr_img: np.ndarray
) -> tuple[np.ndarray, int]:
    
    MIN_AREA = 3000
    MAX_AREA = (ref_img.shape[0] * ref_img.shape[1]) * 0.8 

    if ref_img is None or aligned_curr_img is None:
        raise ValueError("ref_img와 aligned_curr_img는 None일 수 없습니다.")
    if ref_img.shape[:2] != aligned_curr_img.shape[:2]:
        raise ValueError("이미지 크기가 일치해야 합니다.")

    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(aligned_curr_img, cv2.COLOR_BGR2GRAY)

    gray_ref = cv2.GaussianBlur(gray_ref, (15, 15), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (15, 15), 0)

    _, ssim_map = ssim(gray_ref, gray_curr, win_size=21, data_range=255, full=True)
    
    diff = np.clip(1.0 - ssim_map, 0.0, 1.0)
    diff_u8 = (diff * 255).astype(np.uint8)

    t_otsu, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_eff = max(float(t_otsu), 100.0) 
    _, binary = cv2.threshold(diff_u8, t_eff, 255, cv2.THRESH_BINARY)

    gray_aligned_raw = cv2.cvtColor(aligned_curr_img, cv2.COLOR_BGR2GRAY)
    _, valid_mask = cv2.threshold(gray_aligned_raw, 1, 255, cv2.THRESH_BINARY)
    valid_mask = cv2.erode(valid_mask, np.ones((40, 40), np.uint8), iterations=1)
    binary = cv2.bitwise_and(binary, binary, mask=valid_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, bw, bh))

    return boxes