"""공통 응답 필드 헬퍼."""
from __future__ import annotations

from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: str
    admin_feedback: str | None = None
