"""Student 스키마."""
from __future__ import annotations

from pydantic import BaseModel


class StudentOut(BaseModel):
    id: int
    student_number: str
    name: str
    room_id: int | None = None
