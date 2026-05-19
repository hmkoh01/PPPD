"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AdminShell } from "@/components/admin/AdminShell";
import { Button } from "@/components/ui/Button";
import { FullscreenCameraModal } from "@/components/camera/FullscreenCameraModal";
import { createAdminRoom } from "@/lib/api";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg"];

export default function AdminRoomCreatePage() {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [refImage, setRefImage] = useState<File | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);

  const roomNumberRef = useRef<HTMLInputElement>(null);
  const studentNumberRef = useRef<HTMLInputElement>(null);
  const studentNameRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const setSelectedFile = (file: File | undefined) => {
    if (!file) {
      setRefImage(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
      return;
    }
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("JPG, PNG, WEBP 형식만 지원합니다.");
      setRefImage(null);
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setError(null);
    setRefImage(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleCameraCapture = (blob: Blob) => {
    const file = new File([blob], "ref_image.jpg", { type: "image/jpeg" });
    setSelectedFile(file);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(null);

    const roomNumber = roomNumberRef.current?.value.trim();
    const studentNumber = studentNumberRef.current?.value.trim();
    const studentName = studentNameRef.current?.value.trim();

    if (!roomNumber || !studentNumber || !studentName || !refImage) {
      setError("호실 번호, 학번, 이름, 기준사진을 모두 입력해 주세요.");
      return;
    }

    setSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("room_number", roomNumber);
      formData.append("student_number", studentNumber);
      formData.append("student_name", studentName);
      formData.append("ref_image", refImage);

      await createAdminRoom(formData);
      setSuccess(`${roomNumber}호 등록이 완료되었습니다.`);

      setTimeout(() => router.push("/admin/rooms"), 600);
    } catch (err) {
      setError(err instanceof Error ? err.message : "호실 등록에 실패했습니다.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AdminShell>
      <div className="space-y-4">
        <div>
          <h1 className="text-2xl font-extrabold text-gray-950">호실 등록</h1>
          <p className="mt-0.5 text-xs text-gray-500">
            호실과 학생 정보를 입력하고 기준사진을 등록합니다.
          </p>
        </div>

        {error && (
          <div className="rounded-2xl bg-red-50 px-4 py-2.5 text-sm text-red-700 ring-1 ring-red-100">
            {error}
          </div>
        )}
        {success && (
          <div className="rounded-2xl bg-emerald-50 px-4 py-2.5 text-sm text-emerald-700 ring-1 ring-emerald-100">
            {success}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 rounded-[20px] bg-white p-4 ring-1 ring-gray-100">
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-gray-600">
                호실 번호 <span className="text-red-500">*</span>
              </label>
              <input
                ref={roomNumberRef}
                type="text"
                placeholder="예) 101"
                className="h-10 w-full rounded-xl bg-gray-50 px-3 text-sm ring-1 ring-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={submitting}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-gray-600">
                학번 <span className="text-red-500">*</span>
              </label>
              <input
                ref={studentNumberRef}
                type="text"
                placeholder="예) 2024123456"
                className="h-10 w-full rounded-xl bg-gray-50 px-3 text-sm ring-1 ring-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={submitting}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-gray-600">
                이름 <span className="text-red-500">*</span>
              </label>
              <input
                ref={studentNameRef}
                type="text"
                placeholder="예) 홍길동"
                className="h-10 w-full rounded-xl bg-gray-50 px-3 text-sm ring-1 ring-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={submitting}
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-xs font-medium text-gray-600">
              기준사진 <span className="text-red-500">*</span>
            </label>

            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="secondary"
                size="md"
                fullWidth
                disabled={submitting}
                onClick={() => setCameraOpen(true)}
              >
                카메라
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="md"
                fullWidth
                disabled={submitting}
                onClick={() => fileInputRef.current?.click()}
              >
                파일 업로드
              </Button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => setSelectedFile(event.target.files?.[0])}
              className="hidden"
              disabled={submitting}
            />

            {previewUrl ? (
              <div className="space-y-1">
                <p className="text-xs font-medium text-gray-500">등록된 기준사진</p>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="기준사진 미리보기"
                  className="h-32 w-full rounded-2xl object-cover ring-1 ring-gray-100"
                />
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center rounded-2xl bg-gray-50 px-4 text-center text-xs text-gray-400 ring-1 ring-gray-100">
                카메라 또는 파일 업로드로 기준사진을 등록해 주세요.
              </div>
            )}
          </div>

          <Button type="submit" variant="primary" size="md" fullWidth disabled={submitting}>
            {submitting ? "등록 중..." : "호실 등록"}
          </Button>
        </form>

        <FullscreenCameraModal
          open={cameraOpen}
          title="기준사진 카메라"
          onClose={() => setCameraOpen(false)}
          onUse={handleCameraCapture}
        />
      </div>
    </AdminShell>
  );
}
