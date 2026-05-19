"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { FullscreenCameraModal } from "@/components/camera/FullscreenCameraModal";
import { getInspection, retakeIssue, submitInspection, uploadIssueCloseup } from "@/lib/api";
import { getDormitorySession, updateDormitorySession } from "@/lib/session";
import { resolveImageUrl } from "@/lib/image";
import type { DormitorySession, Inspection, Issue } from "@/lib/types";

function hasBbox(issue: Issue): boolean {
  return (
    Number.isFinite(issue.x) &&
    Number.isFinite(issue.y) &&
    Number.isFinite(issue.width) &&
    Number.isFinite(issue.height) &&
    issue.width > 0 &&
    issue.height > 0
  );
}

function isIssueComplete(issue: Issue): boolean {
  return Boolean(issue.closeup_image_url && issue.vlm_reason);
}

function hasCloseup(issue: Issue): boolean {
  return Boolean(issue.closeup_image_url);
}

export default function StudentIssuesPage() {
  const router = useRouter();
  const [session, setSession] = useState<DormitorySession | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [selectedIssueId, setSelectedIssueId] = useState<number | null>(null);
  const [issuesLoading, setIssuesLoading] = useState(true);
  const [issuesError, setIssuesError] = useState<string | null>(null);
  const [isMapOpen, setIsMapOpen] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [uploadingIssueId, setUploadingIssueId] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [retakingIssueId, setRetakingIssueId] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const loadIssues = useCallback((inspectionId: number) => {
    setIssuesLoading(true);
    setIssuesError(null);
    getInspection(inspectionId)
      .then((insp) => {
        const nextIssues = insp.issues ?? [];
        setInspection(insp);
        setIssues(nextIssues);
        setSelectedIssueId((current) => {
          if (current && nextIssues.some((issue) => issue.id === current)) return current;
          return nextIssues.find((issue) => !isIssueComplete(issue))?.id ?? nextIssues[0]?.id ?? null;
        });
      })
      .catch(() => setIssuesError("클로즈업 촬영 목록을 불러오지 못했어요."))
      .finally(() => setIssuesLoading(false));
  }, []);

  useEffect(() => {
    const s = getDormitorySession();
    if (!s) {
      router.replace("/student/login");
      return;
    }

    setSession(s);
    if (s.inspectionId) {
      loadIssues(s.inspectionId);
    } else {
      setIssuesLoading(false);
    }
  }, [loadIssues, router]);

  useEffect(() => {
    if (!isMapOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isMapOpen]);

  const selectedIssue = useMemo(
    () => issues.find((issue) => issue.id === selectedIssueId) ?? null,
    [issues, selectedIssueId],
  );
  const selectedIndex = selectedIssue ? issues.findIndex((issue) => issue.id === selectedIssue.id) : -1;
  const completedCount = issues.filter(isIssueComplete).length;
  const remainingCount = Math.max(issues.length - completedCount, 0);
  const canSubmit = !issuesLoading && issues.length > 0 && remainingCount === 0 && !uploadingIssueId;
  const finalImageUrl = resolveImageUrl(inspection?.final_image_url);
  const canOverlay = Boolean(finalImageUrl && !imageError && naturalSize && issues.some(hasBbox));
  const selectedIsAnalyzing = Boolean(
    selectedIssue && (uploadingIssueId === selectedIssue.id || (hasCloseup(selectedIssue) && !selectedIssue.vlm_reason)),
  );
  const selectedIsComplete = Boolean(selectedIssue && isIssueComplete(selectedIssue));

  const updateIssue = (updated: Issue) => {
    setIssues((prev) => prev.map((issue) => (issue.id === updated.id ? updated : issue)));
  };

  const handleSelectIssue = (issueId: number) => {
    setSelectedIssueId((current) => (current === issueId ? null : issueId));
    setUploadError(null);
    setSubmitError(null);
  };

  const handleCapture = async (blob: Blob) => {
    if (!selectedIssue) return;

    setUploadingIssueId(selectedIssue.id);
    setUploadError(null);
    setSubmitError(null);

    try {
      const res = await uploadIssueCloseup(selectedIssue.id, blob);
      updateIssue({
        ...selectedIssue,
        ...res.issue,
        closeup_image_url:
          res.closeup_image_url ?? res.issue.closeup_image_url ?? selectedIssue.closeup_image_url ?? null,
      });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "사진 업로드에 실패했어요.");
    } finally {
      setUploadingIssueId(null);
    }
  };

  const handleRetake = async () => {
    if (!selectedIssue) return;

    setRetakingIssueId(selectedIssue.id);
    setUploadError(null);
    setSubmitError(null);

    try {
      const updated = await retakeIssue(selectedIssue.id);
      updateIssue(updated);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "다시 촬영을 시작하지 못했어요.");
    } finally {
      setRetakingIssueId(null);
    }
  };

  const handleSubmit = async () => {
    if (!session?.inspectionId) return;
    if (!canSubmit) {
      setSubmitError("모든 영역을 촬영하면 제출할 수 있어요.");
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    try {
      const statusRes = await submitInspection(session.inspectionId);
      updateDormitorySession({ status: statusRes.status });
      router.push("/student/result");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "제출에 실패했어요.");
    } finally {
      setSubmitting(false);
    }
  };

  if (!session) return null;

  return (
    <StudentShell title="클로즈업 촬영" back={{ label: "처음으로", href: "/" }} trailing={`${session.roomNumber}호`}>
      <StudentStepper currentStep={4} />

      {issuesLoading ? (
        <div className="space-y-3">
          <div className="h-36 animate-pulse rounded-[24px] bg-white ring-1 ring-gray-100" />
          <div className="h-14 animate-pulse rounded-2xl bg-white ring-1 ring-gray-100" />
        </div>
      ) : issuesError ? (
        <div className="space-y-3 rounded-[24px] bg-red-50 p-4 text-sm text-red-700 ring-1 ring-red-100">
          <p>{issuesError}</p>
          <Button
            variant="secondary"
            size="md"
            fullWidth
            onClick={() => session.inspectionId && loadIssues(session.inspectionId)}
          >
            다시 불러오기
          </Button>
        </div>
      ) : (
        <>
          <section className="space-y-5 rounded-[24px] bg-white p-5 ring-1 ring-gray-100">
            <div className="space-y-2">
              <h1 className="text-3xl font-extrabold text-gray-950">
                {canSubmit ? "클로즈업 촬영 완료" : "클로즈업 촬영"}
              </h1>
              <p className="text-sm leading-relaxed text-gray-500">
                {canSubmit
                  ? "모든 영역을 촬영했어요. 관리자가 확인할 수 있도록 제출해 주세요."
                  : `${issues.length}개 영역을 촬영해 주세요.`}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-2xl bg-blue-50 px-4 py-3 ring-1 ring-blue-100">
                <p className="text-2xl font-extrabold text-blue-700">{completedCount}</p>
                <p className="mt-0.5 text-xs font-bold text-blue-700">완료</p>
              </div>
              <div className="rounded-2xl bg-gray-50 px-4 py-3 ring-1 ring-gray-100">
                <p className="text-2xl font-extrabold text-gray-900">{remainingCount}</p>
                <p className="mt-0.5 text-xs font-bold text-gray-500">남음</p>
              </div>
            </div>

            <Button
              variant="primary"
              size="lg"
              fullWidth
              onClick={() => {
                setSelectedIssueId(null);
                setIsMapOpen(true);
              }}
            >
              {canSubmit ? "손상 영역 다시 보기" : "손상 영역 보기"}
            </Button>

            {!canSubmit && (
              <p className="text-center text-xs font-semibold text-gray-400">
                모든 영역을 촬영하면 제출할 수 있어요.
              </p>
            )}
          </section>

          {submitError && (
            <div className="rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
              {submitError}
            </div>
          )}

          {canSubmit && (
            <Button variant="primary" size="lg" fullWidth disabled={submitting} onClick={handleSubmit}>
              {submitting ? "제출 중..." : "제출하기"}
            </Button>
          )}
        </>
      )}

      {mounted &&
        isMapOpen &&
        createPortal(
          <div className="fixed left-0 top-0 z-50 h-[100dvh] w-screen overflow-hidden bg-[#F7F8FA] text-gray-950">
          <div
            className="absolute inset-0 flex items-center justify-center overflow-hidden"
            onClick={() => setSelectedIssueId(null)}
          >
            {finalImageUrl && !imageError ? (
              <div className="relative flex max-h-full max-w-full items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={finalImageUrl}
                  alt="퇴사 사진"
                  onError={() => setImageError(true)}
                  onLoad={(event) =>
                    setNaturalSize({
                      width: event.currentTarget.naturalWidth,
                      height: event.currentTarget.naturalHeight,
                    })
                  }
                  className="block max-h-[100dvh] max-w-screen object-contain"
                />

                {canOverlay &&
                  issues.filter(hasBbox).map((issue, index) => {
                    const selected = issue.id === selectedIssueId;
                    const complete = isIssueComplete(issue);
                    const analyzing = hasCloseup(issue) && !issue.vlm_reason;
                    const uploading = issue.id === uploadingIssueId;
                    const left = (issue.x / naturalSize!.width) * 100;
                    const top = (issue.y / naturalSize!.height) * 100;
                    const width = (issue.width / naturalSize!.width) * 100;
                    const height = (issue.height / naturalSize!.height) * 100;

                    return (
                      <button
                        key={issue.id}
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          handleSelectIssue(issue.id);
                        }}
                        aria-label={`${index + 1}번 의심 영역 선택`}
                        className={[
                          "absolute min-h-9 min-w-9 rounded-lg border-2 shadow-lg transition-colors",
                          selected
                            ? "border-blue-400 bg-blue-500/35"
                            : complete
                            ? "border-emerald-400 bg-emerald-400/15"
                            : analyzing || uploading
                            ? "border-blue-300 bg-blue-400/20"
                            : "border-amber-300 bg-amber-400/18",
                        ].join(" ")}
                        style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                      >
                        <span className="absolute -left-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-white text-xs font-extrabold text-gray-900 shadow ring-1 ring-black/10">
                          {uploading || analyzing ? (
                            <span className="h-3 w-3 animate-spin rounded-full border-2 border-blue-100 border-t-blue-600" />
                          ) : (
                            index + 1
                          )}
                        </span>
                      </button>
                    );
                  })}
              </div>
            ) : (
              <div className="mx-5 rounded-3xl bg-white/90 px-6 py-5 text-center text-sm text-gray-600 shadow-2xl ring-1 ring-white/70 backdrop-blur-xl">
                퇴사 사진을 불러오지 못했어요.
              </div>
            )}
          </div>

          <header className="absolute left-0 right-0 top-0 z-30 px-4 pt-[calc(env(safe-area-inset-top)+12px)]">
            <div className="flex items-center justify-between gap-3">
              <button
                type="button"
                className="rounded-full bg-white/90 px-3 py-2 text-sm font-bold text-gray-950 shadow-lg backdrop-blur-xl ring-1 ring-white/70"
                onClick={() => setIsMapOpen(false)}
                aria-label="요약 화면으로 돌아가기"
              >
                돌아가기
              </button>
              <span className="rounded-full bg-white/90 px-3 py-2 text-xs font-extrabold text-gray-950 shadow-lg backdrop-blur-xl ring-1 ring-white/70">
                {completedCount}/{issues.length} 완료
              </span>
            </div>
          </header>

          {selectedIssue && (
            <div className="absolute inset-x-4 bottom-[calc(env(safe-area-inset-bottom)+16px)] z-30 mx-auto max-w-lg rounded-3xl border border-white/70 bg-white/90 p-3 text-gray-950 shadow-2xl backdrop-blur-xl">
              {selectedIsAnalyzing ? (
                <div className="flex items-center justify-center gap-3 py-1.5">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-blue-100 border-t-blue-600" />
                  <span className="text-sm font-bold text-gray-700">사진을 확인하고 있어요</span>
                </div>
              ) : selectedIsComplete ? (
                <div className="space-y-2.5">
                  <p className="text-sm font-extrabold tracking-tight text-gray-950">
                    {selectedIndex + 1}번 영역 촬영 완료
                  </p>
                  <div>
                    <p className="text-xs font-extrabold text-blue-600">AI 확인 결과</p>
                    <p className="mt-1 max-h-10 overflow-hidden text-sm leading-5 text-gray-600">
                      {selectedIssue.vlm_reason}
                    </p>
                  </div>
                  <p className="text-xs font-semibold text-gray-500">최종 판단은 관리자가 검토합니다.</p>
                  {uploadError && <p className="rounded-2xl bg-red-50 px-3 py-2 text-xs text-red-700">{uploadError}</p>}
                  <Button
                    variant="secondary"
                    size="md"
                    fullWidth
                    disabled={retakingIssueId === selectedIssue.id}
                    onClick={handleRetake}
                  >
                    {retakingIssueId === selectedIssue.id ? "준비 중..." : "다시 촬영"}
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <p className="min-w-0 flex-1 text-sm font-extrabold text-gray-950">{selectedIndex + 1}번 영역</p>
                  <Button
                    variant="primary"
                    size="md"
                    disabled={uploadingIssueId === selectedIssue.id}
                    onClick={() => setIsCameraOpen(true)}
                    className="h-11 shrink-0 rounded-2xl px-5 text-base"
                  >
                    촬영하기
                  </Button>
                  {uploadError && <p className="sr-only">{uploadError}</p>}
                </div>
              )}
            </div>
          )}

          <FullscreenCameraModal
            open={isCameraOpen}
            title="클로즈업 촬영"
            onClose={() => setIsCameraOpen(false)}
            onUse={handleCapture}
          />
          </div>,
          document.body,
        )}
    </StudentShell>
  );
}


