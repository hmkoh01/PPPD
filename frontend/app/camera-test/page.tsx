"use client";

import { useState } from "react";
import { CameraCapture, CameraMode } from "@/components/camera/CameraCapture";
import { blobToObjectUrl, revokeObjectUrl } from "@/lib/image";

const MODES: CameraMode[] = ["checkin", "checkout", "closeup"];

export default function CameraTestPage() {
  const [mode, setMode] = useState<CameraMode>("checkin");
  const [showCamera, setShowCamera] = useState(false);
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null);
  const [capturedSize, setCapturedSize] = useState<number>(0);

  const handleCapture = (blob: Blob) => {
    if (capturedUrl) revokeObjectUrl(capturedUrl);
    const url = blobToObjectUrl(blob);
    setCapturedUrl(url);
    setCapturedSize(blob.size);
    setShowCamera(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-sm mx-auto space-y-4">
        {/* 개발용 페이지 배너 */}
        <div className="bg-yellow-50 border border-yellow-300 rounded-xl px-3 py-2 text-xs text-yellow-800 text-center">
          개발용 테스트 페이지입니다. 실제 서비스에서는 사용하지 마세요.
        </div>
        <h1 className="text-lg font-bold text-gray-900">카메라 테스트</h1>

        {/* 모드 선택 */}
        <div className="flex gap-2">
          {MODES.map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex-1 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                mode === m
                  ? "bg-indigo-600 text-white border-indigo-600"
                  : "bg-white text-gray-600 border-gray-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {/* 카메라 */}
        {showCamera ? (
          <CameraCapture
            mode={mode}
            onCapture={handleCapture}
            onCancel={() => setShowCamera(false)}
          />
        ) : (
          <button
            onClick={() => setShowCamera(true)}
            className="w-full bg-indigo-600 text-white py-3 rounded-2xl text-sm font-semibold"
          >
            카메라 열기 ({mode})
          </button>
        )}

        {/* 촬영 결과 */}
        {capturedUrl && (
          <div className="space-y-2">
            <p className="text-xs text-gray-500 font-medium">
              촬영 결과 ({(capturedSize / 1024).toFixed(1)} KB)
            </p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={capturedUrl}
              alt="촬영 결과"
              className="w-full rounded-2xl aspect-video object-cover bg-gray-100"
            />
          </div>
        )}
      </div>
    </div>
  );
}
