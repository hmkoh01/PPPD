# 기숙사 퇴사 점검 플랫폼 — FastAPI 백엔드

## 요구사항

- Python 3.10+
- pip

---

## 설치 및 초기 설정

```bash
# 1. backend/ 디렉터리로 이동
cd backend

# 2. 가상환경 생성 및 활성화
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 파일 생성
cp .env.example .env
# .env 를 열어 GEMINI_API_KEY 등 실제 값을 입력하세요
```

---

## DB 초기화

**처음 실행 전, 또는 모델 변경 후 반드시 실행합니다.**

```bash
# backend/ 디렉터리에서 실행
cd backend
python -m app.db.init_db
```

성공 시 출력 예시:
```
DB 초기화 시작...
완료: 테이블 및 data/images/ 디렉터리가 준비되었습니다.
DB 파일: sqlite:///./data/db.sqlite
```

생성되는 파일 및 디렉터리:
```
backend/
├── data/
│   ├── db.sqlite      ← SQLite DB 파일
│   └── images/        ← 이미지 저장 디렉터리
```

---

## 서버 실행

```bash
# backend/ 디렉터리에서 실행
cd backend
uvicorn app.main:app --reload --port 8000
```

---

## API 엔드포인트 확인

| URL | 설명 |
|---|---|
| `GET  http://localhost:8000/health` | 헬스 체크 → `{"status": "ok"}` |
| `GET  http://localhost:8000/docs`   | Swagger UI (자동 생성) |
| `GET  http://localhost:8000/redoc`  | ReDoc |
| `GET  http://localhost:8000/images/{filename}` | 저장된 이미지 서빙 |

---

## DB 스키마 개요

```
rooms
  id (PK)  room_number (unique)  status  created_at  updated_at

students
  id (PK)  student_number (unique)  name  room_id (FK→rooms.id)  created_at  updated_at

inspections
  id (PK)  room_id (FK)  student_id (FK)  status
  ref_image_path  initial_image_path  final_image_path
  admin_feedback  submitted_at  reviewed_at  created_at  updated_at

issues
  id (PK)  inspection_id (FK)  x  y  width  height  status
  crop_image_path  closeup_image_path  vlm_reason  created_at  updated_at
```

상태값:
- Room / Inspection: `ready` → `checked_in` → `pending_review` → `approved` / `rejected`
- Issue: `red` (미검증) → `orange` (의심) / `green` (이상없음)

---

## 기존 Streamlit 앱과의 대응 관계

| 구 Streamlit | 신 FastAPI |
|---|---|
| `room_id` (문자열 PK) | `rooms.room_number` (unique string) |
| `rooms["status"]` | `rooms.status` + `inspections.status` |
| `rooms["issues"]` (JSON 배열) | `issues` 테이블 (정규화) |
| `box_coords: [x,y,w,h]` | `issues.x`, `y`, `width`, `height` |
| `st.secrets["GEMINI_API_KEY"]` | `.env` → `os.environ["GEMINI_API_KEY"]` |

---

## Storage Service (이미지 저장/로드)

### 역할

`app/services/storage_service.py` 가 모든 이미지 파일 I/O 를 담당합니다.

| 기능 | 함수 |
|---|---|
| UploadFile 저장 | `await save_upload_file(file, image_type, ...)` |
| bytes 저장 | `save_image_bytes(bytes, image_type, ...)` |
| BGR ndarray 저장 | `save_ndarray(ndarray, image_type, ...)` |
| 파일 로드 → BGR | `load_image_bgr(path)` |
| bytes → BGR | `decode_image_bytes_to_bgr(bytes)` |
| BGR → JPEG bytes | `encode_bgr_to_jpeg_bytes(ndarray)` |
| public URL 생성 | `get_public_image_url(path_or_filename)` |

### 기존 Streamlit db_utils 함수 대응

| 구 Streamlit (utils/db_utils.py) | 신 FastAPI (storage_service.py) |
|---|---|
| `save_image_bytes(room_id, type, data)` | `save_image_bytes(data, type, room_number=room_id)` |
| `save_image_ndarray(room_id, type, img)` | `save_ndarray(img, type, room_number=room_id)` |
| `load_image(path)` | `load_image_bgr(path)` |

### 이미지 저장 경로와 URL 규칙

```
디스크 경로: backend/data/images/{filename}
DB 저장값:  filename (상대 경로)
Public URL: http://localhost:8000/images/{filename}
```

파일명 형식:
```
{prefix}_{image_type}_{YYYYMMDD_HHMMSS}_{uuid6}.jpg

예:
  101_initial_20260513_153012_ab12cd.jpg       ← room_number 기준
  inspection_3_final_20260513_153012_ab12cd.jpg ← inspection_id 기준
  issue_7_closeup_20260513_153012_ab12cd.jpg    ← issue_id 기준
```

이미지 타입 (`ImageType` 상수):
| 상수 | 값 | 설명 |
|---|---|---|
| `ImageType.REF` | `"ref"` | 관리자 등록 기준 사진 |
| `ImageType.INITIAL` | `"initial"` | 학생 Step1 입사 초기 사진 |
| `ImageType.FINAL` | `"final"` | Step2 최종 사진 (정합 전) |
| `ImageType.ALIGNED_FINAL` | `"aligned_final"` | ORB 정합 완료 최종 사진 |
| `ImageType.CROP` | `"crop"` | 이슈 박스 자동 크롭 |
| `ImageType.CLOSEUP` | `"closeup"` | 학생 Step3 클로즈업 사진 |

### 수동 테스트 방법

**1. 서버 기동 후 Swagger UI 에서 이미지 업로드 테스트**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs 에서 POST /api/rooms 등 실행
```

**2. Python REPL 에서 직접 확인**

```python
# backend/ 디렉터리에서 실행
from app.services.storage_service import save_image_bytes, load_image_bgr, get_public_image_url
import cv2, numpy as np

# ndarray 저장 테스트
img = np.zeros((100, 100, 3), dtype=np.uint8)  # 검은 100x100 이미지
from app.services.storage_service import save_ndarray
from app.core.constants import ImageType

result = save_ndarray(img, ImageType.INITIAL, room_number="101")
print(result.filename)     # 101_initial_20260513_...jpg
print(result.public_url)   # http://localhost:8000/images/101_initial_...jpg
print(result.size_bytes)   # ~600 bytes 수준

# 로드 테스트
loaded = load_image_bgr(result.file_path)
print(loaded.shape)        # (100, 100, 3)

# URL 확인
print(get_public_image_url(result.filename))
```

**3. 저장된 이미지 URL 접근 확인**

서버 기동 후:
```
GET http://localhost:8000/images/{저장된_파일명}
```

---

## Gemini Service (VLM 클로즈업 분석)

### 역할

`app/services/gemini_service.py` 가 Gemini VLM 분석 전체를 담당합니다.
기존 Streamlit `2_Student.py` 의 `vlm_analyze()` 를 FastAPI 서비스 레이어로 분리했습니다.

**API 키는 절대 프론트엔드에 노출되지 않습니다. 항상 서버 환경변수에서만 읽습니다.**

### 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_API_KEY` | `""` | Gemini API 키 (필수, aistudio.google.com 에서 발급) |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | 사용할 Gemini 모델명 |

### 주요 타입

```python
@dataclass
class GeminiAnalysisResult:
    result: Literal["clean", "suspicious"]
    reason: str                  # 한국어 판단 사유
    raw_text: str | None = None  # Gemini 원문 응답
```

### 예외 계층

```
GeminiServiceError (RuntimeError)
├── GeminiConfigError   — GEMINI_API_KEY 미설정 (fallback 없이 전파)
└── GeminiAnalysisError — API 호출 / JSON 파싱 실패 (내부에서 fallback 처리)
```

### Fallback 정책

다음 상황에서는 예외를 발생시키는 대신 `result="suspicious"` 를 반환합니다:

| 상황 | 동작 |
|---|---|
| 이미지 로드/인코딩 실패 | suspicious fallback |
| Gemini API 호출 오류 (네트워크, 할당량 등) | suspicious fallback |
| 응답 JSON 파싱 실패 | suspicious fallback |
| result 값이 clean/suspicious 가 아님 | suspicious fallback |
| reason 필드가 비어 있음 | 기본 사유로 대체 |
| **GEMINI_API_KEY 미설정** | **GeminiConfigError 전파** |

Fallback reason: `"AI 분석 결과를 안정적으로 해석하지 못해 관리자 확인이 필요합니다."`

### 사용 예시

```python
# 동기 호출 (일반 Python 코드에서)
from app.services.gemini_service import analyze_closeup, GeminiConfigError

try:
    result = analyze_closeup(image_path="issue_7_closeup_20260513_153012_ab12cd.jpg")
    print(result.result)   # "clean" | "suspicious"
    print(result.reason)   # 한국어 판단 사유
except GeminiConfigError as e:
    print(f"API 키 설정 필요: {e}")

# 이미지 bytes 로 직접 전달
result = analyze_closeup(image_bytes=raw_bytes)

# async 호출 (FastAPI endpoint 에서)
from app.services.gemini_service import analyze_closeup_async

@router.post("/issues/{issue_id}/analyze")
async def analyze_issue(issue_id: int, db: Session = Depends(get_db)):
    issue = db.get(Issue, issue_id)
    result = await analyze_closeup_async(image_path=issue.closeup_image_path)
    # ... issue.vlm_reason = result.reason; issue.status = ...
```

### 동기/비동기

`analyze_closeup()` 은 동기 함수입니다. Gemini Python SDK 가 동기 API 이기 때문입니다.

- **FastAPI endpoint** → `await analyze_closeup_async(...)` 사용 (내부에서 `asyncio.to_thread` 로 실행)
- **일반 Python 코드** → `analyze_closeup(...)` 직접 사용

TODO: httpx 기반 비동기 Gemini 클라이언트로 교체하면 네이티브 async 가 가능합니다.

### Phase 6 API 연결 시 주의점

1. `GeminiConfigError` 는 500 Internal Server Error 로 매핑하세요.
2. fallback 결과(`suspicious`)도 정상 200 응답이므로 `result.raw_text` 로 구분합니다.
3. `analyze_closeup_async` 를 `await` 하면 되므로 endpoint 코드가 간결합니다.

---

## Vision Service (이미지 정합 및 차이 검출)

### 역할

`app/services/vision_service.py` 가 CV 파이프라인 전체를 담당합니다.

| 단계 | 내용 |
|---|---|
| 1 | initial 이미지 로드 (`load_image_bgr`) |
| 2 | final 이미지 디코딩 (bytes 또는 파일 경로) |
| 3 | ORB 특징점 정합 (`align_images`) |
| 4 | 정합 결과 저장 (`aligned_final`) |
| 5 | SSIM 차이 검출 (`detect_difference`) |
| 6 | 이슈 박스 크롭 이미지 저장 |
| 7 | `VisionDetectionResult` 반환 |

### 주요 타입

```python
@dataclass
class DetectedIssue:
    x: int; y: int; width: int; height: int
    crop_image_path: str | None  # DB 저장용 파일명

@dataclass
class VisionDetectionResult:
    aligned_final_image_path: str   # DB 저장용 파일명
    issues: list[DetectedIssue]
```

### 예외 계층

```
VisionServiceError (RuntimeError)
├── VisionAlignmentFailed  — ORB 정합 실패
└── VisionDetectionFailed  — SSIM 차이 검출 실패
```

### 사용 예시

```python
from app.services.vision_service import align_and_detect, VisionAlignmentFailed

try:
    result = align_and_detect(
        initial_image_path="101_initial_20260513_153012_ab12cd.jpg",
        final_image_bytes=raw_bytes,   # UploadFile 에서 읽은 bytes
        room_number="101",
        inspection_id=3,
    )
    print(result.aligned_final_image_path)  # DB 저장용 파일명
    for issue in result.issues:
        print(issue.x, issue.y, issue.width, issue.height, issue.crop_image_path)
except VisionAlignmentFailed as e:
    print(f"정합 실패: {e}")  # 학생에게 재촬영 요청
```

### CV 알고리즘 파라미터 (vision_utils.py)

| 항목 | 값 | 비고 |
|---|---|---|
| ORB nfeatures | 2000 | |
| Lowe 비율 테스트 | 0.75 | |
| 최소 good matches | 15 | 미달 시 `ImageAlignmentError` |
| RANSAC reproj 임계값 | 5.0 px | |
| GaussianBlur kernel | (15, 15) | SSIM 전처리 |
| SSIM win_size | 21 | |
| Otsu 최소 threshold | 100 | |
| valid_mask erode | (40, 40) | warpPerspective 검은 테두리 제거 |
| 형태학 연산 kernel | (7, 7) rect | open 3회 + close 3회 |
| 최소 윤곽선 면적 | 3000 px² | |
| 최대 윤곽선 면적 | 이미지 면적 × 80% | |

---

## API 전체 플로우 & Swagger 테스트 가이드

### 서버 기동

```bash
cd backend
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs  → Swagger UI
```

### 전체 플로우 테스트 순서 (Swagger UI 기준)

#### 1단계 — 관리자: 호실 등록

```
POST /api/admin/rooms  (multipart/form-data)
  room_number:    "101"
  student_number: "2024123456"
  student_name:   "홍길동"
  ref_image:      <파일 선택>

응답: { room, student, inspection, ref_image_url }
  → inspection.id 를 메모해 두세요 (이후 모든 단계에서 사용)
```

#### 2단계 — 학생: 인증

```
POST /api/students/verify
  { "student_number": "2024123456", "name": "홍길동" }

응답: { student, room, inspection, status }
  → inspection.id 확인
```

#### 3단계 — 학생: 입사 초기 사진 업로드

```
POST /api/inspections/{inspection_id}/initial-image  (multipart/form-data)
  file: <초기 사진>

응답: { inspection_id, status, initial_image_path, initial_image_url }
  → status = "checked_in"
```

#### 4단계 — 학생: 퇴사 최종 사진 업로드 + CV 분석

```
POST /api/inspections/{inspection_id}/final-image  (multipart/form-data)
  file: <최종 사진>

응답: { inspection, final_image_url, issues }
  → issues 배열에 각 이슈의 id, x/y/w/h, crop_image_url 포함
  → VisionAlignmentFailed → HTTP 422 (재촬영 안내)
```

#### 5단계 — 학생: 이슈 클로즈업 촬영 + Gemini 분석

```
POST /api/issues/{issue_id}/closeup  (multipart/form-data)
  file: <클로즈업 사진>

응답: { issue, closeup_image_url, result, reason }
  → issue.status = "green"(clean) | "orange"(suspicious)
  → Gemini 오류 시에도 HTTP 200 + status="orange" (fallback 정책)
```

재촬영이 필요한 경우:
```
PATCH /api/issues/{issue_id}/retake
  → issue.status = "red" 로 초기화
```

#### 6단계 — 학생: 최종 제출

```
POST /api/inspections/{inspection_id}/submit
  → 미검증(red) 이슈 있으면 400
  → 성공 시 status = "pending_review"
```

#### 7단계 — 관리자: 점검 목록 확인

```
GET /api/admin/inspections?status_filter=pending_review
GET /api/admin/inspections/{inspection_id}
```

#### 8단계 — 관리자: 승인 또는 반려

```
PATCH /api/admin/inspections/{inspection_id}/approve
  { "admin_feedback": "이상 없음" }   ← 선택사항

PATCH /api/admin/inspections/{inspection_id}/reject
  { "admin_feedback": "청소 후 재제출 바람" }   ← 필수
```

### 이미지 URL 규칙

| 항목 | 값 |
|---|---|
| DB 저장 | filename (예: `101_initial_20260513_153012_ab12cd.jpg`) |
| Public URL | `http://localhost:8000/images/{filename}` |
| 브라우저 접근 | `GET http://localhost:8000/images/{filename}` |

모든 응답의 `*_image_url` 필드는 바로 `<img src>` 에 사용 가능합니다.

### Status 전환표

```
Room / Inspection 상태 전환:
  ready
    ↓ initial-image 업로드
  checked_in
    ↓ submit
  pending_review
    ↓ approve       ↓ reject
  approved        rejected
                    ↓ (학생 재촬영 후 다시 submit)
                  pending_review

Issue 상태 전환:
  red (미검증)
    ↓ closeup 업로드 + Gemini 분석
  green (clean) | orange (suspicious)
    ↓ retake
  red (재촬영)
```

### CV 실패 422 정책

| 예외 | HTTP | 안내 메시지 |
|---|---|---|
| `VisionAlignmentFailed` | 422 | 같은 위치와 구도로 다시 촬영해 주세요 |
| `VisionDetectionFailed` | 422 | CV 분석 실패 |
| `VisionServiceError` | 422 | CV 분석 실패 |
| 그 외 예외 | 500 | 서버 오류 |

### Gemini Fallback 정책 (GeminiConfigError 포함 모든 오류)

| 상황 | 동작 |
|---|---|
| API 키 미설정 (`GeminiConfigError`) | HTTP 200 + `status=orange`, `reason="AI 설정 오류로 관리자 확인이 필요합니다."` |
| 네트워크 오류, JSON 파싱 실패 등 | HTTP 200 + `status=orange`, fallback reason |

**이유**: 시연 안정성 최우선. Gemini 오류로 학생 플로우 전체가 막히지 않도록 설계.
관리자가 orange 이슈를 직접 확인하여 승인/반려합니다.

---

## 시연 준비 (Demo Reset)

### 전체 데이터 초기화 (수동)

> 데이터를 완전히 초기화하려면 다음 순서로 수동 삭제합니다.

```bash
# 1. backend 서버 중지 후 실행

# 2. DB 삭제
rm backend/data/db.sqlite   # macOS/Linux
del backend\data\db.sqlite  # Windows

# 3. 이미지 삭제
rm -rf backend/data/images/  # macOS/Linux
rd /s /q backend\data\images\ # Windows

# 4. DB 재초기화
cd backend
python -m app.db.init_db

# 5. 서버 재시작
uvicorn app.main:app --reload --port 8000
```

### 테스트 호실 등록 (Swagger UI)

서버 시작 후 http://localhost:8000/docs 에서:

```
POST /api/admin/rooms (multipart/form-data)
  room_number:    "101"
  student_number: "2024000001"
  student_name:   "테스트학생"
  ref_image:      (기준 사진 JPG/PNG)
```

### 빠른 확인 URL

| 역할 | URL |
|---|---|
| Swagger API 문서 | http://localhost:8000/docs |
| 헬스 체크 | http://localhost:8000/health |
| 관리자 화면 | http://localhost:3000/admin |
| 학생 화면 | http://localhost:3000/student |
| 관리자 — 호실 등록 | http://localhost:3000/admin/rooms |
| 관리자 — 점검 리뷰 | http://localhost:3000/admin/reviews |
| 카메라 테스트 (개발용) | http://localhost:3000/camera-test |

---

## 알려진 제한사항

- **SQLite** — 동시 쓰기 불가. 단일 시연 환경에 적합.
- **HTTPS 필요** — 모바일 카메라는 HTTPS 또는 localhost에서만 동작.
- **Gemini API 키 미설정 시** — closeup 분석이 항상 `orange`(suspicious) fallback 처리됨. 관리자가 직접 판단 필요.
- **ORB 정합 실패** — 입사/퇴사 사진 구도가 크게 다르면 422 오류. 같은 위치/각도에서 재촬영 필요.

## 남은 TODO (Phase 13+)

- Canny edge overlay — 카메라 뷰파인더에 기준 사진 edge 실시간 렌더링
- closeup 촬영 시 이슈 위치 포커스 가이드 UI
- Gemini SDK `google-generativeai` → `google-genai` 마이그레이션
- JWT/세션 인증 추가
- PostgreSQL 전환
- S3/Supabase Storage 연동

---

## 기존 Streamlit 앱 병행 실행

백엔드와 Streamlit 앱은 독립적으로 실행됩니다.

```bash
# 프로젝트 루트(code/)에서 실행
streamlit run pages/1_Admin.py    # 관리자 앱 (포트 8501)
streamlit run pages/2_Student.py  # 학생 앱  (포트 8502)
```
