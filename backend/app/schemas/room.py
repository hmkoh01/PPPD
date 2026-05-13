"""Room 스키마."""
from __future__ import annotations

from pydantic import BaseModel

from app.schemas.student import StudentOut
from app.schemas.inspection import InspectionOut


class RoomOut(BaseModel):
    id: int
    room_number: str
    status: str


class RoomDetailOut(RoomOut):
    student: StudentOut | None = None
    latest_inspection: InspectionOut | None = None
    ref_image_url: str | None = None


class AdminCreateRoomResponse(BaseModel):
    room: RoomOut
    student: StudentOut
    inspection: InspectionOut
    ref_image_url: str | None = None
