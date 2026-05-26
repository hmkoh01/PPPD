"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { CameraCapture } from "@/components/camera/CameraCapture";
import { getDormitorySession, updateDormitorySession } from "@/lib/session";
import { checkImageAlignment, uploadInitialImage } from "@/lib/api";
import { resolveImageUrl } from "@/lib/image";
import type { DormitorySession } from "@/lib/types";

const COPY = {
  title: "\uc785\uc0ac \ucd2c\uc601",
  back: "\ucc98\uc74c\uc73c\ub85c",
  room: "\ubc30\uc815 \ud638\uc2e4",
  reference: "\uae30\uc900 \uc0ac\uc9c4",
  referenceAlt: "\uae30\uc900 \uc0ac\uc9c4",
  referenceFallback: "\uae30\uc900 \uc0ac\uc9c4\uc744 \ubd88\ub7ec\uc62c \uc218 \uc5c6\uc5b4\uc694.",
  cameraTitle: "\uc785\uc0ac \uc0ac\uc9c4 \ucd2c\uc601",
  cameraDescription: "\uae30\uc900 \uc0ac\uc9c4\uacfc \uac19\uc740 \uad6c\ub3c4\ub85c \ucd2c\uc601\ud574 \uc8fc\uc138\uc694.",
  uploadErrorMissing: "\uc810\uac80 ID\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. \ub2e4\uc2dc \uc778\uc99d\ud574 \uc8fc\uc138\uc694.",
  uploadError: "\uc0ac\uc9c4 \uc5c5\ub85c\ub4dc\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4.",
  uploading: "\uc0ac\uc9c4\uc744 \uc5c5\ub85c\ub4dc\ud558\ub294 \uc911\uc785\ub2c8\ub2e4.",
};

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
    setSession(s);
  }, [router]);

  if (!session) return null;

  const refUrl = resolveImageUrl(session.refImageUrl);

  const handleCapture = async (blob: Blob) => {
    if (!session.inspectionId) {
      setUploadError(COPY.uploadErrorMissing);
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
      setUploadError(err instanceof Error ? err.message : COPY.uploadError);
      setUploading(false);
    }
  };

  const handleValidateCapture = async (blob: Blob) => {
    if (!session.inspectionId) {
      return { ok: false, status: "poor" as const, message: COPY.uploadErrorMissing };
    }
    try {
      const res = await checkImageAlignment(session.inspectionId, blob, "checkin");
      return {
        ok: res.ok,
        score: res.score,
        status: res.status,
        message: res.ok ? undefined : res.message,
      };
    } catch {
      return {
        ok: false,
        status: "poor" as const,
        message: "구도를 다시 한 번 맞춰볼게요.",
      };
    }
  };

  const handleAnalyzeFrame = async (blob: Blob) => {
    if (!session.inspectionId) {
      return { ok: false, status: "poor" as const };
    }
    const res = await checkImageAlignment(session.inspectionId, blob, "checkin", true);
    return {
      ok: res.ok,
      score: res.score,
      status: res.status,
      message: res.message,
    };
  };

  return (
    <StudentShell
      title={COPY.title}
      back={{ label: COPY.back, href: "/" }}
      trailing={`${session.roomNumber}\ud638`}
    >
      <StudentStepper currentStep={2} />

      <div className="rounded-2xl bg-white px-4 py-3 text-sm leading-relaxed text-gray-600 ring-1 ring-gray-100">
        {COPY.cameraDescription}
      </div>

      <section className="space-y-2">
        <div className="flex items-center justify-between text-xs font-semibold text-gray-500">
          <span>{COPY.reference}</span>
        </div>
        {refUrl && !refImgErr ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={refUrl}
            alt={COPY.referenceAlt}
            onError={() => setRefImgErr(true)}
            className="aspect-[16/9] max-h-44 w-full rounded-[20px] bg-gray-100 object-cover ring-1 ring-gray-100"
          />
        ) : (
          <div className="flex aspect-[16/9] max-h-44 items-center justify-center rounded-[20px] bg-gray-100 text-sm text-gray-400">
            {COPY.referenceFallback}
          </div>
        )}
      </section>

      {uploading ? (
        <div className="flex flex-col items-center gap-3 rounded-[24px] bg-white px-4 py-7 ring-1 ring-gray-100">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-100 border-t-blue-600" />
          <p className="text-sm text-gray-500">{COPY.uploading}</p>
        </div>
      ) : (
        <CameraCapture
          mode="checkin"
          title={COPY.cameraTitle}
          description={COPY.cameraDescription}
          overlayImageUrl={refUrl ?? undefined}
          onValidateCapture={handleValidateCapture}
          onAnalyzeFrame={handleAnalyzeFrame}
          onCapture={handleCapture}
        />
      )}

      {uploadError && (
        <div className="rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
          {uploadError}
        </div>
      )}
    </StudentShell>
  );
}
