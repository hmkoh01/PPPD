"use client";

import { useEffect, useRef, useState } from "react";
import { AdminShell } from "@/components/admin/AdminShell";
import { RoomTable } from "@/components/admin/RoomTable";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { getAdminRooms, createAdminRoom } from "@/lib/api";
import type { Room } from "@/lib/types";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg"];

export default function AdminRoomsPage() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const roomNumberRef = useRef<HTMLInputElement>(null);
  const studentNumberRef = useRef<HTMLInputElement>(null);
  const studentNameRef = useRef<HTMLInputElement>(null);
  const refImageRef = useRef<HTMLInputElement>(null);

  const loadRooms = () => {
    setLoading(true);
    getAdminRooms()
      .then(setRooms)
      .catch((err) => setError(err instanceof Error ? err.message : "불러오기 실패"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadRooms();
  }, []);

  const handleFileChange = () => {
    const file = refImageRef.current?.files?.[0];
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("JPG, PNG, WEBP 이미지만 업로드할 수 있습니다.");
      refImageRef.current!.value = "";
      setPreviewUrl(null);
      return;
    }
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    const roomNumber = roomNumberRef.current?.value.trim();
    const studentNumber = studentNumberRef.current?.value.trim();
    const studentName = studentNameRef.current?.value.trim();
    const file = refImageRef.current?.files?.[0];

    if (!roomNumber || !studentNumber || !studentName || !file) {
      setError("모든 필드와 기준 사진을 입력해 주세요.");
      return;
    }
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("JPG, PNG, WEBP 이미지만 업로드할 수 있습니다.");
      return;
    }

    setSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("room_number", roomNumber);
      formData.append("student_number", studentNumber);
      formData.append("student_name", studentName);
      // backend field name: ref_image (UploadFile = File(...))
      formData.append("ref_image", file);

      await createAdminRoom(formData);
      setSuccess(`${roomNumber}호 등록이 완료되었습니다.`);

      // 폼 초기화
      if (roomNumberRef.current) roomNumberRef.current.value = "";
      if (studentNumberRef.current) studentNumberRef.current.value = "";
      if (studentNameRef.current) studentNameRef.current.value = "";
      if (refImageRef.current) refImageRef.current.value = "";
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);

      loadRooms();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "등록에 실패했습니다.";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AdminShell>
      <div className="space-y-6">
        <h1 className="text-xl font-bold text-gray-900">호실 관리</h1>

        {/* 등록 폼 */}
        <Card
          title="새 호실 등록"
          description="호실 번호, 학생 정보, 기준 사진을 입력하세요."
        >
          <form onSubmit={handleSubmit} className="space-y-4 mt-2">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  호실 번호 <span className="text-red-500">*</span>
                </label>
                <input
                  ref={roomNumberRef}
                  type="text"
                  placeholder="예) 101"
                  className="w-full px-3 py-2 rounded-xl border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  disabled={submitting}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  학번 <span className="text-red-500">*</span>
                </label>
                <input
                  ref={studentNumberRef}
                  type="text"
                  placeholder="예) 2024123456"
                  className="w-full px-3 py-2 rounded-xl border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  disabled={submitting}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  이름 <span className="text-red-500">*</span>
                </label>
                <input
                  ref={studentNameRef}
                  type="text"
                  placeholder="예) 홍길동"
                  className="w-full px-3 py-2 rounded-xl border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  disabled={submitting}
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                기준 사진 <span className="text-red-500">*</span>{" "}
                <span className="text-gray-400">(JPG, PNG, WEBP)</span>
              </label>
              <input
                ref={refImageRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
                className="w-full text-sm text-gray-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
                disabled={submitting}
              />
              {previewUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="미리보기"
                  className="mt-2 h-32 rounded-xl object-cover border border-gray-200"
                />
              )}
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}
            {success && (
              <div className="bg-green-50 border border-green-200 rounded-xl px-4 py-3 text-sm text-green-700">
                {success}
              </div>
            )}

            <Button
              type="submit"
              variant="primary"
              size="md"
              disabled={submitting}
            >
              {submitting ? "등록 중…" : "등록하기"}
            </Button>
          </form>
        </Card>

        {/* 호실 목록 */}
        <div className="bg-white rounded-2xl border border-gray-100 p-4">
          <h2 className="text-sm font-semibold text-gray-800 mb-4">
            전체 호실{!loading && ` (${rooms.length}개)`}
          </h2>
          {loading ? (
            <div className="space-y-2">
              {[...Array(4)].map((_, i) => (
                <div
                  key={i}
                  className="h-10 bg-gray-100 rounded-xl animate-pulse"
                />
              ))}
            </div>
          ) : (
            <RoomTable rooms={rooms} />
          )}
        </div>
      </div>
    </AdminShell>
  );
}
