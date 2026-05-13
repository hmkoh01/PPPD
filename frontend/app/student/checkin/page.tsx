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
import { uploadInitialImage } from "@/lib/api";
import { resolveImageUrl } from "@/lib/image";
import type { DormitorySession } from "@/lib/types";

export default function StudentCheckinPage() {
  const router = useRouter();
  const [session, setSession] = useState<DormitorySession | null>(null);
  const [refImgErr, setRefImgErr] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    const s = getDormitorySession();
    if (!s) {
      router.replace("/student/login");
      return;
    }
    if (s.status !== "ready") {
      router.replace(statusToPath(s.status));
      return;
    }
    setSession(s);
  }, [router]);

  if (!session) return null;

  const handleCapture = async (blob: Blob) => {
    if (!session.inspectionId) {
      setUploadError("점검 ID를 찾을 수 없습니다. 다시 로그인해 주세요.");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const res = await uploadInitialImage(session.inspectionId, blob);
      updateDormitorySession({
        status: "checked_in",
        initialImageUrl: res.initial_image_url ?? null,
      });
      router.push("/student/checkout");
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "업로드에 실패했습니다."
      );
      setUploading(false);
    }
  };

  const refUrl = resolveImageUrl(session.refImageUrl);

  return (
    <StudentShell title="입사 사진 촬영" back={{ label: "처음으로", href: "/student" }}>
      <StudentStepper currentStep={1} />

      <Card title="입사 초기 사진 촬영">
        <p className="text-sm text-gray-600 leading-relaxed">
          기준 사진의 구도에 맞춰 방 전체를 촬영해 주세요.
          이 사진은 퇴사 때 비교 기준으로 사용됩니다.
        </p>
      </Card>

      {/* 호실 정보 */}
      <div className="bg-indigo-50 rounded-xl px-4 py-3 text-sm">
        <span className="text-indigo-600 font-medium">배정 호실: </span>
        <span className="text-indigo-900 font-bold">{session.roomNumber}호</span>
      </div>

      {/* 기준 사진 미리보기 */}
      <div className="space-y-1">
        <p className="text-xs text-gray-500 font-medium">기준 사진 (관리자 등록)</p>
        {refUrl && !refImgErr ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={refUrl}
            alt="기준 사진"
            onError={() => setRefImgErr(true)}
            className="w-full rounded-2xl object-cover aspect-video bg-gray-100"
          />
        ) : (
          <div className="bg-gray-100 rounded-2xl aspect-video flex flex-col items-center justify-center gap-2 text-gray-400">
            <span className="text-3xl">🖼️</span>
            <p className="text-sm">기준 사진 없음</p>
          </div>
        )}
      </div>

      {/* 카메라 */}
      {uploading ? (
        <div className="bg-gray-50 rounded-2xl px-4 py-6 flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
          <p className="text-sm text-gray-500">업로드 중…</p>
        </div>
      ) : (
        <CameraCapture
          mode="checkin"
          overlayImageUrl={refUrl ?? undefined}
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
