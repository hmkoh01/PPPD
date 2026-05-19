"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { CameraCapture } from "@/components/camera/CameraCapture";
import { getDormitorySession, updateDormitorySession } from "@/lib/session";
import { submitInspection, uploadFinalImage } from "@/lib/api";
import { resolveImageUrl } from "@/lib/image";
import type { DormitorySession } from "@/lib/types";

const COPY = {
  title: "\ud1f4\uc0ac \ucd2c\uc601",
  back: "\ucc98\uc74c\uc73c\ub85c",
  room: "\ubc30\uc815 \ud638\uc2e4",
  reference: "\uc785\uc0ac \uc0ac\uc9c4",
  referenceAlt: "\uc785\uc0ac \uc0ac\uc9c4",
  referenceFallback: "\uc785\uc0ac \uc0ac\uc9c4\uc744 \ubd88\ub7ec\uc62c \uc218 \uc5c6\uc5b4\uc694.",
  cameraTitle: "\ud1f4\uc0ac \uc0ac\uc9c4 \ucd2c\uc601",
  cameraDescription: "\uc785\uc0ac \uc0ac\uc9c4\uacfc \uac19\uc740 \uad6c\ub3c4\ub85c \ud604\uc7ac \uc0c1\ud0dc\ub97c \ucd2c\uc601\ud574 \uc8fc\uc138\uc694.",
  feedback: "\uad00\ub9ac\uc790 \ud53c\ub4dc\ubc31",
  uploadErrorMissing: "\uc810\uac80 ID\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. \ub2e4\uc2dc \uc778\uc99d\ud574 \uc8fc\uc138\uc694.",
  uploadError: "\uc0ac\uc9c4 \uc5c5\ub85c\ub4dc\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4.",
  uploading: "\uc0ac\uc9c4\uc744 \ubd84\uc11d\ud558\ub294 \uc911\uc785\ub2c8\ub2e4.",
};

function toCheckoutError(err: unknown): string {
  if (!(err instanceof Error)) return COPY.uploadError;
  return err.message;
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
    setSession(s);
  }, [router]);

  if (!session) return null;

  const isRejected = session.status === "rejected";
  const initUrl = resolveImageUrl(session.initialImageUrl);

  const handleCapture = async (blob: Blob) => {
    if (!session.inspectionId) {
      setUploadError(COPY.uploadErrorMissing);
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const res = await uploadFinalImage(session.inspectionId, blob);
      updateDormitorySession({ finalImageUrl: res.final_image_url ?? null });

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

  return (
    <StudentShell
      title={COPY.title}
      back={{ label: COPY.back, href: "/" }}
      trailing={`${session.roomNumber}\ud638`}
    >
      <StudentStepper currentStep={3} />

      <div className="rounded-2xl bg-white px-4 py-3 text-sm leading-relaxed text-gray-600 ring-1 ring-gray-100">
        {COPY.cameraDescription}
      </div>

      {isRejected && session.adminFeedback && (
        <div className="rounded-[20px] bg-red-50 px-4 py-3 ring-1 ring-red-100">
          <p className="mb-1 text-xs font-semibold text-red-700">{COPY.feedback}</p>
          <p className="text-sm leading-relaxed text-red-800">{session.adminFeedback}</p>
        </div>
      )}

      <section className="space-y-2">
        <div className="flex items-center justify-between text-xs font-semibold text-gray-500">
          <span>{COPY.reference}</span>
        </div>
        {initUrl && !initImgErr ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={initUrl}
            alt={COPY.referenceAlt}
            onError={() => setInitImgErr(true)}
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
          mode="checkout"
          title={COPY.cameraTitle}
          description={COPY.cameraDescription}
          overlayImageUrl={initUrl ?? undefined}
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
