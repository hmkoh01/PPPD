"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { Card } from "@/components/ui/Card";
import { CameraCapture } from "@/components/camera/CameraCapture";
import { getIssueStatusLabel, getIssueStatusTone } from "@/lib/status";
import {
  getInspection,
  submitInspection,
  uploadIssueCloseup,
  retakeIssue,
} from "@/lib/api";
import { getDormitorySession, updateDormitorySession } from "@/lib/session";
import { resolveImageUrl } from "@/lib/image";
import type { DormitorySession, Issue } from "@/lib/types";

// ── 이슈 카드 ─────────────────────────────────────────────

function IssueCard({
  issue,
  index,
  onUploaded,
}: {
  issue: Issue;
  index: number;
  onUploaded: (updated: Issue) => void;
}) {
  const [cropErr, setCropErr] = useState(false);
  const [closeupErr, setCloseupErr] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [retaking, setRetaking] = useState(false);

  const cropUrl = resolveImageUrl(issue.crop_image_url);
  const closeupUrl = resolveImageUrl(issue.closeup_image_url);

  const handleCapture = async (blob: Blob) => {
    setShowCamera(false);
    setUploading(true);
    setUploadError(null);
    try {
      const res = await uploadIssueCloseup(issue.id, blob);
      onUploaded({ ...issue, ...res.issue, closeup_image_url: res.closeup_image_url ?? null });
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "업로드에 실패했습니다."
      );
    } finally {
      setUploading(false);
    }
  };

  const handleRetake = async () => {
    setRetaking(true);
    setUploadError(null);
    try {
      const updated = await retakeIssue(issue.id);
      onUploaded(updated);
      setCloseupErr(false);
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "재촬영 초기화에 실패했습니다."
      );
    } finally {
      setRetaking(false);
    }
  };

  return (
    <div className="border border-gray-200 rounded-2xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-800">영역 {index}</span>
        <Badge tone={getIssueStatusTone(issue.status)}>
          {getIssueStatusLabel(issue.status)}
        </Badge>
      </div>

      <p className="text-xs text-gray-400">
        위치: ({issue.x}, {issue.y}) / 크기: {issue.width}×{issue.height}px
      </p>

      {/* 크롭 이미지 */}
      {cropUrl && !cropErr ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={cropUrl}
          alt={`영역 ${index} 크롭`}
          onError={() => setCropErr(true)}
          className="w-full rounded-xl object-cover aspect-video bg-gray-100"
        />
      ) : (
        <div className="bg-gray-100 rounded-xl aspect-video flex items-center justify-center text-gray-400 text-sm">
          크롭 이미지 없음
        </div>
      )}

      {/* VLM 분석 결과 */}
      {issue.vlm_reason && (
        <p className="text-xs text-gray-600 bg-gray-50 rounded-lg px-2 py-1.5 leading-relaxed">
          AI: {issue.vlm_reason}
        </p>
      )}

      {/* red → 클로즈업 촬영 필요 */}
      {issue.status === "red" && (
        <>
          {showCamera ? (
            <CameraCapture
              mode="closeup"
              overlayImageUrl={cropUrl ?? undefined}
              onCapture={handleCapture}
              onCancel={() => setShowCamera(false)}
            />
          ) : uploading ? (
            <div className="bg-gray-50 rounded-xl py-4 flex flex-col items-center gap-2">
              <div className="w-5 h-5 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
              <p className="text-xs text-gray-500">AI가 가까이 찍은 사진을 확인하고 있어요…</p>
            </div>
          ) : (
            <Button
              variant="primary"
              size="sm"
              fullWidth
              onClick={() => setShowCamera(true)}
            >
              클로즈업 촬영하기
            </Button>
          )}
        </>
      )}

      {/* orange → 결과 표시 + 선택적 재촬영 */}
      {issue.status === "orange" && (
        <>
          {closeupUrl && !closeupErr && (
            <div className="space-y-1">
              <p className="text-xs text-gray-500 font-medium">클로즈업 사진</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={closeupUrl}
                alt={`영역 ${index} 클로즈업`}
                onError={() => setCloseupErr(true)}
                className="w-full rounded-xl object-cover aspect-video bg-gray-100"
              />
            </div>
          )}
          <p className="text-xs text-orange-600 bg-orange-50 rounded-lg px-2 py-1.5">
            관리자가 최종 확인합니다. 더 명확하게 촬영하려면 다시 촬영하세요.
          </p>
          {retaking ? (
            <div className="bg-gray-50 rounded-xl py-3 flex items-center justify-center gap-2">
              <div className="w-4 h-4 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
              <p className="text-xs text-gray-500">초기화 중…</p>
            </div>
          ) : (
            <Button variant="secondary" size="sm" fullWidth onClick={handleRetake}>
              다시 촬영하기
            </Button>
          )}
        </>
      )}

      {/* green → 완료 */}
      {issue.status === "green" && closeupUrl && !closeupErr && (
        <div className="space-y-1">
          <p className="text-xs text-gray-500 font-medium">클로즈업 사진</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={closeupUrl}
            alt={`영역 ${index} 클로즈업`}
            onError={() => setCloseupErr(true)}
            className="w-full rounded-xl object-cover aspect-video bg-gray-100"
          />
        </div>
      )}

      {uploadError && (
        <p className="text-xs text-red-600">{uploadError}</p>
      )}
    </div>
  );
}

// ── 메인 페이지 ──────────────────────────────────────────

export default function StudentIssuesPage() {
  const router = useRouter();
  const [session, setSession] = useState<DormitorySession | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [issuesLoading, setIssuesLoading] = useState(true);
  const [issuesError, setIssuesError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const loadIssues = useCallback((inspectionId: number) => {
    setIssuesLoading(true);
    setIssuesError(null);
    getInspection(inspectionId)
      .then((insp) => setIssues(insp.issues ?? []))
      .catch(() => setIssuesError("이슈 목록을 불러오지 못했습니다."))
      .finally(() => setIssuesLoading(false));
  }, []);

  useEffect(() => {
    const s = getDormitorySession();
    if (!s) {
      router.replace("/student/login");
      return;
    }
    // checked_in 또는 rejected(재점검 중) 상태만 허용
    if (s.status !== "checked_in" && s.status !== "rejected") {
      if (s.status === "ready") {
        router.replace("/student/checkin");
      } else {
        router.replace("/student/result");
      }
      return;
    }
    setSession(s);
    if (s.inspectionId) {
      loadIssues(s.inspectionId);
    } else {
      setIssuesLoading(false);
    }
  }, [router, loadIssues]);

  const handleIssueUploaded = (updated: Issue) => {
    setIssues((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
  };

  const handleSubmit = async () => {
    if (!session?.inspectionId) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const statusRes = await submitInspection(session.inspectionId);
      updateDormitorySession({ status: statusRes.status });
      router.push("/student/result");
    } catch (err) {
      setSubmitError(
        err instanceof Error ? err.message : "제출에 실패했습니다."
      );
    } finally {
      setSubmitting(false);
    }
  };

  if (!session) return null;

  const redCount = issues.filter((i) => i.status === "red").length;
  const canSubmit = !issuesLoading && redCount === 0;

  return (
    <StudentShell
      title="차이 영역 확인"
      back={{ label: "퇴사 촬영으로", href: "/student/checkout" }}
    >
      <StudentStepper currentStep={3} />

      {issuesLoading ? (
        <div className="space-y-3">
          {[...Array(2)].map((_, i) => (
            <div key={i} className="h-40 bg-gray-100 rounded-2xl animate-pulse" />
          ))}
        </div>
      ) : issuesError ? (
        <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
          {issuesError}
          <button
            className="ml-2 underline"
            onClick={() => session.inspectionId && loadIssues(session.inspectionId)}
          >
            다시 시도
          </button>
        </div>
      ) : issues.length === 0 ? (
        <>
          <Card title="AI 감지 결과">
            <p className="text-sm text-gray-600">
              AI가 입사 때와 다른 부분을 찾지 못했습니다. 바로 제출할 수 있습니다.
            </p>
          </Card>
          {submitError && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
              {submitError}
            </div>
          )}
          <Button
            variant="primary"
            size="lg"
            fullWidth
            disabled={submitting}
            onClick={handleSubmit}
          >
            {submitting ? "제출 중…" : "최종 제출하기"}
          </Button>
        </>
      ) : (
        <>
          <Card title="AI 감지 결과">
            <p className="text-sm text-gray-600 leading-relaxed">
              AI가 입사 때와 다른 부분을{" "}
              <strong>{issues.length}곳</strong> 발견했습니다.
              {redCount > 0 && (
                <span className="text-orange-600">
                  {" "}클로즈업 촬영이 필요한 영역이 {redCount}건 있습니다.
                </span>
              )}
            </p>
          </Card>

          <div className="space-y-3">
            {issues.map((issue, idx) => (
              <IssueCard
                key={issue.id}
                issue={issue}
                index={idx + 1}
                onUploaded={handleIssueUploaded}
              />
            ))}
          </div>

          {submitError && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
              {submitError}
            </div>
          )}

          <Button
            variant="primary"
            size="lg"
            fullWidth
            disabled={!canSubmit || submitting}
            onClick={handleSubmit}
          >
            {submitting
              ? "제출 중…"
              : canSubmit
              ? "최종 제출하기"
              : `클로즈업 촬영 필요 (${redCount}건)`}
          </Button>
        </>
      )}
    </StudentShell>
  );
}
