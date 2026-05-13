"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { Card } from "@/components/ui/Card";
import { CameraCapture } from "@/components/camera/CameraCapture";
import {
  getDormitorySession,
  statusToPath,
  updateDormitorySession,
} from "@/lib/session";
import { uploadFinalImage, submitInspection } from "@/lib/api";
import { resolveImageUrl } from "@/lib/image";
import type { DormitorySession } from "@/lib/types";

/** API 오류 → 사용자 친화적 메시지 */
function toCheckoutError(err: unknown): string {
  if (!(err instanceof Error)) return "업로드에 실패했습니다.";
  const msg = err.message;
  if (msg.includes("연결할 수 없")) return "서버와 연결하지 못했어요. 잠시 후 다시 시도해 주세요.";
  if (msg.includes("정합 실패") || msg.includes("alignment")) {
    return "사진 구도가 달라 비교가 어려워요. 같은 위치에서 입사 사진과 동일한 각도로 다시 촬영해 주세요.";
  }
  if (msg.includes("CV 분석 실패") || msg.includes("422")) {
    return "사진 분석 중 오류가 발생했습니다. 다시 촬영해 주세요.";
  }
  if (msg.includes("초기 사진")) return msg; // 백엔드 메시지 그대로 표시 (Step 1 미완료 안내)
  return msg;
}

export default function StudentCheckoutPage() {
  const router = useRouter();
  const [session, setSession] = useState<DormitorySession | null>(null);
  const [initImgErr, setInitImgErr] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    const s = getDormitorySession();
    if (!s) {
      router.replace("/student/login");
      return;
    }
    // checked_in 또는 rejected 상태만 허용
    if (s.status !== "checked_in" && s.status !== "rejected") {
      router.replace(statusToPath(s.status));
      return;
    }
    setSession(s);
  }, [router]);

  if (!session) return null;

  const isRejected = session.status === "rejected";

  const handleCapture = async (blob: Blob) => {
    if (!session.inspectionId) {
      setUploadError("점검 ID를 찾을 수 없습니다. 다시 로그인해 주세요.");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const res = await uploadFinalImage(session.inspectionId, blob);
      updateDormitorySession({ finalImageUrl: res.final_image_url ?? null });

      // issues가 있으면 클로즈업 페이지로, 없으면 바로 제출
      if (res.issues && res.issues.length > 0) {
        router.push("/student/issues");
      } else {
        const statusRes = await submitInspection(session.inspectionId);
        updateDormitorySession({ status: statusRes.status });
        router.push("/student/result");
      }
    } catch (err) {
      setUploadError(toCheckoutError(err));
      setUploading(false);
    }
  };

  const initUrl = resolveImageUrl(session.initialImageUrl);

  return (
    <StudentShell
      title="퇴사 사진 촬영"
      back={{ label: "처음으로", href: "/student" }}
    >
      <StudentStepper currentStep={2} />

      {/* 재점검 요청 시 관리자 피드백 표시 */}
      {isRejected && session.adminFeedback && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3">
          <p className="text-xs font-semibold text-red-700 mb-1">재점검 요청 사유</p>
          <p className="text-sm text-red-800 leading-relaxed">{session.adminFeedback}</p>
        </div>
      )}

      <Card title="퇴사 전 현재 상태 촬영">
        <p className="text-sm text-gray-600 leading-relaxed">
          입사 때 찍은 초기 사진과 동일한 각도에서 현재 방 상태를 촬영해
          주세요. AI가 두 사진을 비교하여 달라진 부분을 자동으로 찾습니다.
        </p>
      </Card>

      {/* 호실 정보 */}
      <div className="bg-indigo-50 rounded-xl px-4 py-3 text-sm">
        <span className="text-indigo-600 font-medium">배정 호실: </span>
        <span className="text-indigo-900 font-bold">{session.roomNumber}호</span>
      </div>

      {/* 입사 초기 사진 미리보기 */}
      {initUrl && (
        <div className="space-y-1">
          <p className="text-xs text-gray-500 font-medium">입사 초기 사진 (비교 기준)</p>
          {!initImgErr ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={initUrl}
              alt="입사 초기 사진"
              onError={() => setInitImgErr(true)}
              className="w-full rounded-2xl object-cover aspect-video bg-gray-100"
            />
          ) : (
            <div className="bg-gray-100 rounded-2xl aspect-video flex items-center justify-center text-gray-400 text-sm">
              사진 로딩 실패
            </div>
          )}
        </div>
      )}

      {/* 카메라 */}
      {uploading ? (
        <div className="bg-gray-50 rounded-2xl px-4 py-6 flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
          <p className="text-sm text-gray-500">사진의 차이점을 확인하고 있어요…</p>
          <p className="text-xs text-gray-400">잠시만 기다려 주세요</p>
        </div>
      ) : (
        <CameraCapture
          mode="checkout"
          overlayImageUrl={initUrl ?? undefined}
          onCapture={handleCapture}
        />
      )}

      {uploadError && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
          {uploadError}
        </div>
      )}
    </StudentShell>
  );
}
