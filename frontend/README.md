# Frontend — 기숙사 퇴사 점검 플랫폼

Next.js 15 + React 19 + TypeScript + Tailwind CSS로 구현된 프론트엔드.

## 기술 스택

- **Next.js 15** (App Router)
- **React 19**
- **TypeScript 5** (strict mode)
- **Tailwind CSS 3**

## 디렉토리 구조

```
frontend/
├── app/
│   ├── page.tsx                  # 랜딩 페이지 (학생/관리자 선택)
│   ├── layout.tsx                # 루트 레이아웃
│   ├── globals.css               # Tailwind 기본 스타일
│   ├── student/
│   │   ├── page.tsx              # 학생 홈
│   │   ├── login/page.tsx        # 학생 인증 (학번+이름)
│   │   ├── checkin/page.tsx      # Step 1: 입사 사진 촬영
│   │   ├── checkout/page.tsx     # Step 2: 퇴사 사진 촬영 + AI 분석
│   │   ├── issues/page.tsx       # Step 3: 차이 영역 클로즈업 촬영
│   │   └── result/page.tsx       # Step 4/5: 점검 결과
│   └── admin/
│       ├── page.tsx              # 관리자 대시보드
│       ├── rooms/page.tsx        # 호실 등록/관리
│       └── reviews/page.tsx      # 점검 리뷰 (승인/반려)
├── components/
│   ├── ui/
│   │   ├── Button.tsx            # variant: primary|secondary|danger|ghost
│   │   ├── Card.tsx              # 제목, 설명, children
│   │   ├── Badge.tsx             # tone: gray|yellow|blue|green|red|orange
│   │   └── Modal.tsx             # ESC 키 지원
│   ├── student/
│   │   ├── StudentShell.tsx      # 모바일 레이아웃 (sticky 헤더, back 링크)
│   │   ├── StudentStepper.tsx    # 5단계 진행 표시
│   │   └── StudentStatusCard.tsx # 상태별 아이콘/메시지/관리자 피드백
│   ├── admin/
│   │   ├── AdminShell.tsx        # 사이드바(데스크톱) + 하단 탭(모바일)
│   │   ├── AdminKpiCards.tsx     # 5개 KPI 카드 (rooms 기반)
│   │   ├── RoomTable.tsx         # 호실 목록 테이블 (썸네일 포함)
│   │   ├── ReviewList.tsx        # 점검 목록 (호실/학생 정보 포함)
│   │   └── InspectionDetail.tsx  # 점검 상세 패널 (이미지, 이슈, 승인/반려)
│   └── camera/
│       └── README.md             # Phase 8 카메라 구현 가이드
└── lib/
    ├── types.ts                  # TypeScript 인터페이스 (백엔드 스키마 대응)
    ├── api.ts                    # apiFetch 래퍼 + 15개 API 함수
    ├── status.ts                 # 상태 레이블/Badge 톤 헬퍼
    └── image.ts                  # resolveImageUrl 헬퍼
```

## 개발 서버 실행

```bash
cd frontend
npm install
npm run dev
```

기본 포트: `http://localhost:3000`

백엔드 API는 `http://localhost:8000`을 기대합니다. 다른 주소를 사용한다면 `lib/api.ts`의 `API_BASE` 상수를 수정하세요.

## 학생 플로우

```
/ (랜딩)
└─ /student              (홈 — 시작하기 버튼)
   └─ /student/login     (학번+이름 입력 → localStorage 세션 저장)
      ├─ /student/checkin   (Step 1: 입사 사진 촬영 — Phase 8 구현)
      ├─ /student/checkout  (Step 2: 퇴사 사진 촬영 — Phase 8 구현)
      ├─ /student/issues    (Step 3: 차이 영역 클로즈업 — Phase 8 구현)
      └─ /student/result    (Step 4/5: 결과 확인)
```

세션 정보는 `localStorage["dormitory_session"]`에 JSON으로 저장됩니다.

## 관리자 플로우

```
/admin           대시보드 (KPI + 호실 목록 요약)
/admin/rooms     호실 등록 폼 + 전체 목록
/admin/reviews   점검 리뷰 (필터링, 승인/반려 모달)
```

## 관리자 화면 API 연결 완료 (Phase 8)

### 사전 요구사항

```bash
# 백엔드 실행 (backend/ 디렉토리에서)
python -m app.db.init_db   # 최초 1회
uvicorn app.main:app --reload --port 8000
```

### 테스트 A — 호실 등록

1. `http://localhost:3000/admin/rooms` 접속
2. 호실 번호 / 학번 / 이름 / 기준 사진(JPG·PNG·WEBP) 입력
3. "등록하기" 클릭
4. 성공 메시지 + 목록 갱신 확인
5. 중복 호실번호나 학번 → 오류 메시지 확인

### 테스트 B — 리뷰 목록

1. Swagger(`http://localhost:8000/docs`) 또는 학생 API로 점검 데이터 생성
2. `http://localhost:3000/admin/reviews` 접속
3. 상태 필터 탭(검토 대기 / 승인 완료 / 재점검 요청 / 전체) 동작 확인
4. 목록 항목에 호실번호·학생이름·학번 표시 확인
5. 항목 클릭 → 우측/하단 상세 패널에 이미지·이슈 표시 확인

### 테스트 C — 승인/반려

1. `pending_review` 상태 점검 선택
2. 상세 패널 "승인" 클릭 → 상태 `approved` 변경 및 목록 갱신 확인
3. 다른 `pending_review` 점검 선택 → "반려" 클릭
4. 피드백 입력 후 "반려 확정" → 상태 `rejected` 및 피드백 표시 확인

### 환경변수 (선택)

기본 백엔드 주소는 `http://localhost:8000`입니다. 변경하려면:

```bash
# frontend/.env.local 생성
NEXT_PUBLIC_API_BASE_URL=http://your-backend:8000
```

## 학생 화면 API 연결 완료 (Phase 9)

### localStorage 세션 구조

`localStorage["dormitory_session"]` 에 저장되는 `DormitorySession` 필드:

| 필드 | 타입 | 설명 |
|---|---|---|
| `studentId` | number | 학생 DB id |
| `studentNumber` | string | 학번 |
| `studentName` | string | 이름 |
| `roomId` | number | 호실 DB id |
| `roomNumber` | string | 호실 번호 |
| `inspectionId` | number \| null | 현재 점검 id |
| `status` | RoomStatus | 현재 상태 |
| `adminFeedback` | string \| null | 관리자 피드백 |
| `refImageUrl` | string \| null | 기준 사진 URL |
| `initialImageUrl` | string \| null | 입사 사진 URL |
| `finalImageUrl` | string \| null | 퇴사 사진 URL |

세션 유틸: `lib/session.ts` — `getDormitorySession`, `setDormitorySession`, `clearDormitorySession`, `updateDormitorySession`, `hasValidDormitorySession`, `statusToPath`

### status 기반 라우팅 규칙

| status | 이동 경로 |
|---|---|
| `ready` | `/student/checkin` |
| `checked_in` | `/student/checkout` |
| `pending_review` | `/student/result` |
| `approved` | `/student/result` |
| `rejected` | `/student/result` (+ `/student/checkout` 재점검 가능) |

### 테스트 A — 학생 인증 성공

1. 백엔드 실행 → `/admin/rooms`에서 호실+학생 등록
2. `/student/login` 접속 → 학번+이름 입력 → 시작하기
3. `ready` 상태 → `/student/checkin` 이동 확인

### 테스트 B — 인증 실패

1. 틀린 학번 또는 이름 입력
2. "일치하는 배정 정보를 찾을 수 없습니다" 메시지 확인
3. 백엔드 꺼진 상태 → "서버에 연결할 수 없습니다" 메시지 확인

### 테스트 C — status 라우팅

1. Swagger(`/docs`)에서 inspection status를 `pending_review`로 변경
2. `/student` 접속
3. API 상태 확인 후 `/student/result` 자동 이동 확인

### 테스트 D — rejected feedback

1. `/admin/reviews`에서 점검 선택 → 반려 + 피드백 입력
2. `/student` 재접속 → `/student/result` 이동
3. 관리자 피드백 표시 확인
4. "재점검 시작하기" → `/student/checkout` 이동 확인

## 학생 촬영 플로우 테스트 (Phase 11 기준)

### 카메라 사용 조건

- **HTTPS 또는 localhost**에서만 `navigator.mediaDevices`가 동작합니다.
- 개발: `http://localhost:3000` 정상 동작
- 배포: 반드시 HTTPS 필요
- 카메라 단독 테스트: `http://localhost:3000/camera-test`

### 테스트 A — 정상 플로우

1. 백엔드 실행 (`uvicorn app.main:app --reload --port 8000`)
2. 프론트엔드 실행 (`npm run dev`)
3. `/admin/rooms` → 호실 번호 / 학번 / 이름 / 기준 사진 등록
4. `/student/login` → 학번 + 이름 입력 → 시작하기
5. `/student/checkin` → 기준 사진 오버레이 확인 후 촬영 → 이 사진 사용 → 업로드
6. `/student/checkout` → 입사 초기 사진 오버레이 확인 후 촬영 → AI 분석 대기
7. 이슈 있음 → `/student/issues` → red 이슈마다 클로즈업 촬영
8. `/student/issues` → 최종 제출하기
9. `/admin/reviews` → pending_review 점검 확인 → 승인
10. `/student/result` → polling으로 `approved` 상태 확인

### 테스트 B — 반려 플로우

1. `/admin/reviews` → 점검 선택 → 반려 + 피드백 입력
2. `/student/result` → `rejected` + 관리자 피드백 표시 확인
3. 재점검 시작하기 → `/student/checkout` 이동
4. 재촬영 후 재제출

### 테스트 C — CV 분석 실패 (422)

1. 입사 때와 완전히 다른 장소 또는 구도로 퇴사 사진 촬영
2. 오류 메시지: "이미지 정합 실패 — 같은 위치와 구도로 다시 촬영해 주세요."
3. 카메라가 다시 표시되어 재촬영 가능 확인

### 테스트 D — 카메라 오류

1. 브라우저에서 카메라 권한 차단
2. "카메라 접근 권한이 없습니다. 브라우저 설정에서 허용해 주세요." 표시 확인
3. 다시 시도 버튼 동작 확인

### issue closeup 검증 절차

- red 이슈 → "클로즈업 촬영하기" 버튼 → 카메라 + crop 이미지 오버레이
- 촬영 후 Gemini 분석 → green(이상 없음) 또는 orange(관리자 확인 필요)
- orange 이슈 → "다시 촬영하기" 버튼으로 retake 가능 (status red로 초기화)
- 모든 이슈 green/orange이면 "최종 제출하기" 활성화

### 관리자 approve/reject 연동 테스트

- 학생 submit → `/admin/reviews` 에 `pending_review` 로 즉시 등장
- 관리자 승인 → 학생 result 페이지가 12초 간격 polling으로 `approved` 감지
- 관리자 반려 + 피드백 → 학생 result에 `rejected` + 피드백 문구 표시

## Phase 13 TODO

- `EdgeOverlay` — 반투명 이미지 대신 Canny edge 실시간 렌더링 (Canvas / WebGL)
- 클로즈업 촬영 전 이슈 영역 포커스 가이드 UI
