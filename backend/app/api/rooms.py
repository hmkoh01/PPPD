"""
호실(Room) API — DEPRECATED.

Phase 6 이후 호실 관련 API 는 모두 /api/admin/rooms 로 통합되었습니다.
이 라우터는 하위 호환성을 위해 유지하지만 모든 엔드포인트는 501을 반환합니다.
새 코드에서는 /api/admin/rooms 를 사용하세요.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter()

_DEPRECATED = HTTPException(
    status_code=status.HTTP_410_GONE,
    detail="이 엔드포인트는 deprecated 되었습니다. /api/admin/rooms 를 사용하세요.",
)


@router.get("", include_in_schema=False)
def list_rooms():
    raise _DEPRECATED


@router.post("", include_in_schema=False)
def create_room():
    raise _DEPRECATED


@router.get("/{room_id}", include_in_schema=False)
def get_room(room_id: str):
    raise _DEPRECATED


@router.patch("/{room_id}", include_in_schema=False)
def update_room(room_id: str):
    raise _DEPRECATED


@router.post("/{room_id}/ref-image", include_in_schema=False)
async def upload_ref_image(room_id: str):
    raise _DEPRECATED
