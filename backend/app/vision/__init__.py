"""
vision 패키지 공개 인터페이스.

사용법:
    from app.vision import align_images, detect_difference, DiffCandidate, ImageAlignmentError
"""
from app.vision.vision_utils import (
    DiffCandidate,
    AlignmentCheckResult,
    ImageAlignmentError,
    align_images,
    check_alignment_quality,
    detect_difference,
)

__all__ = [
    "DiffCandidate",
    "AlignmentCheckResult",
    "ImageAlignmentError",
    "align_images",
    "check_alignment_quality",
    "detect_difference",
]
