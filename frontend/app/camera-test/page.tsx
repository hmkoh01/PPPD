"use client";

import { useState } from "react";
import { CameraCapture, CameraMode } from "@/components/camera/CameraCapture";
import { blobToObjectUrl, revokeObjectUrl } from "@/lib/image";

const MODES: CameraMode[] = ["checkin", "checkout", "closeup"];

export default function CameraTestPage() {
  const [mode, setMode] = useState<CameraMode>("checkin");
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null);
  const [capturedSize, setCapturedSize] = useState<number>(0);

  const handleCapture = (blob: Blob) => {
    if (capturedUrl) revokeObjectUrl(capturedUrl);
    setCapturedUrl(blobToObjectUrl(blob));
    setCapturedSize(blob.size);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-sm mx-auto space-y-4">
        <h1 className="text-lg font-bold text-gray-900">카메라 테스트</h1>

        <div className="flex gap-2">
          {MODES.map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex-1 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                mode === m
                  ? "border-blue-600 bg-blue-600 text-white"
                  : "bg-white text-gray-600 border-gray-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        <CameraCapture mode={mode} onCapture={handleCapture} />

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
