"""Issue 스키마."""
from __future__ import annotations

from pydantic import BaseModel


class IssueOut(BaseModel):
    id: int
    inspection_id: int
    x: int
    y: int
    width: int
    height: int
    status: str
    candidate_type: str = "small_damage"
    student_note: str | None = None
    vlm_reason: str | None = None
    crop_image_path: str | None = None
    crop_image_url: str | None = None
    closeup_image_path: str | None = None
    closeup_image_url: str | None = None


class CloseupUploadResponse(BaseModel):
    issue: IssueOut
    closeup_image_url: str | None = None
    result: str
    reason: str
