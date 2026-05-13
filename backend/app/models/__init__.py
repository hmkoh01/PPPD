"""
ORM 모델 패키지.

이 __init__.py 에서 모든 모델을 import 하여
Base.metadata 에 테이블 정의가 등록되도록 합니다.
init_db.py 에서 `import app.models` 한 번으로 전체 등록이 완료됩니다.

import 순서 주의:
  Room → Student(room_id FK) → Inspection(room_id, student_id FK) → Issue(inspection_id FK)
"""
from app.models.room import Room                    # noqa: F401
from app.models.student import Student              # noqa: F401
from app.models.inspection import Inspection        # noqa: F401
from app.models.issue import Issue                  # noqa: F401
