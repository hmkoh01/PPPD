"use client";

import { useEffect, useMemo, useState } from "react";
import type { Inspection, Issue, Room } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ImageLightbox } from "@/components/ui/ImageLightbox";
import { getStatusLabel, getStatusTone, getIssueStatusLabel, getIssueStatusTone } from "@/lib/status";
import { resolveImageUrl } from "@/lib/image";

interface InspectionDetailProps {
  inspection: Inspection;
  room: Room | null;
  onApprove: () => void;
  onReject: () => void;
  actionLoading: boolean;
  actionError: string | null;
}

function InspectionImage({
  url,
  label,
}: {
  url: string | null | undefined;
  label: string;
}) {
  const [err, setErr] = useState(false);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const imageUrl = resolveImageUrl(url);
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-gray-500">{label}</p>
      {imageUrl && !err ? (
        <>
          <button type="button" className="block w-full" onClick={() => setLightboxOpen(true)}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl}
              alt={label}
              onError={() => setErr(true)}
              className="aspect-video w-full rounded-[18px] bg-gray-100 object-cover"
            />
          </button>
          <ImageLightbox
            open={lightboxOpen}
            src={imageUrl}
            alt={label}
            onClose={() => setLightboxOpen(false)}
          />
        </>
      ) : (
        <div className="flex aspect-video w-full items-center justify-center rounded-[18px] bg-gray-100 text-sm text-gray-400">
          사진 없음
        </div>
      )}
    </div>
  );
}

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

function getReviewState(issue: Issue): string {
  if (issue.status === "needs_confirmation") return "학생 확인 자료 미제출";
  if (issue.status === "evidence_submitted") return "확인 자료 제출됨";
  if (issue.status === "cleared") return "추가 확인 불필요";
  if (!issue.closeup_image_url) return "학생 확인 자료 미제출";
  if (!issue.vlm_reason) return "AI 참고 분석 중";
  return "자료 제출 완료";
}

function IssueMap({
  finalImageUrl,
  issues,
  selectedIssueId,
  onSelect,
}: {
  finalImageUrl: string | null | undefined;
  issues: Issue[];
  selectedIssueId: number | null;
  onSelect: (issueId: number) => void;
}) {
  const [err, setErr] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const imageUrl = resolveImageUrl(finalImageUrl);
  const canOverlay = Boolean(imageUrl && !err && naturalSize && issues.some(hasBbox));

  return (
    <div className="space-y-3 rounded-[20px] bg-gray-50 p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-bold text-gray-700">AI 감지 후보 (소규모 손상 · 큰 객체 · 재촬영 권장)</p>
        <span className="text-xs font-medium text-gray-400">{issues.length}건</span>
      </div>

      {imageUrl && !err ? (
        <div className="relative overflow-hidden rounded-[18px] bg-gray-100">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl}
            alt="퇴사 최종 사진 의심 영역"
            onError={() => setErr(true)}
            onLoad={(event) => {
              setNaturalSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              });
            }}
            className="block w-full"
          />

          {canOverlay &&
            issues.filter(hasBbox).map((issue, index) => {
              const selected = issue.id === selectedIssueId;
              const completed = Boolean(issue.closeup_image_url);
              const left = (issue.x / naturalSize!.width) * 100;
              const top = (issue.y / naturalSize!.height) * 100;
              const width = (issue.width / naturalSize!.width) * 100;
              const height = (issue.height / naturalSize!.height) * 100;

              return (
                <button
                  key={issue.id}
                  type="button"
                  onClick={() => onSelect(issue.id)}
                  className={[
                    "absolute rounded-lg border-2 text-[10px] font-extrabold shadow-sm transition-colors",
                    selected
                      ? "border-blue-500 bg-blue-500/20 text-blue-700"
                      : completed
                      ? "border-emerald-500 bg-emerald-500/15 text-emerald-700"
                      : "border-amber-500 bg-amber-500/15 text-amber-700",
                  ].join(" ")}
                  style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                  aria-label={`의심 영역 ${index + 1}`}
                >
                  <span className="absolute -left-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-white ring-1 ring-gray-200">
                    {completed ? "✓" : index + 1}
                  </span>
                </button>
              );
            })}
        </div>
      ) : (
        <div className="flex aspect-video w-full items-center justify-center rounded-[18px] bg-gray-100 text-sm text-gray-400">
          퇴사 사진 없음
        </div>
      )}

      {!canOverlay && issues.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-gray-400">좌표 정보가 없어 사진 위에 확인 필요 영역을 표시할 수 없습니다.</p>
        </div>
      )}
    </div>
  );
}

function getCandidateTypeLabel(type: string | null | undefined): string {
  if (type === "large_object") return "큰 객체/방치물 후보";
  if (type === "recapture_recommended") return "재촬영 권장";
  return "확인 필요 영역";
}

function getCandidateTypeContext(type: string | null | undefined): string {
  if (type === "large_object")
    return "큰 물건이나 방치물이 감지된 영역입니다. 학생의 확인 사진과 함께 검토해 주세요.";
  if (type === "recapture_recommended")
    return "이미지 전체에 큰 변화가 감지되었습니다. 사진 구도 차이일 수 있으므로 원본 사진과 함께 확인해 주세요.";
  return "AI가 전후 차이를 감지한 소규모 영역입니다. 학생 확인 자료와 함께 검토해 주세요.";
}

function VlmSummary({ reason }: { reason: string }) {
  const firstLine = reason.split("\n").find((l) => l.trim() !== "") ?? "";
  return (
    <p className="text-xs leading-relaxed text-blue-700">{firstLine}</p>
  );
}

function IssueDetail({ issue, index }: { issue: Issue; index: number }) {
  const [cropErr, setCropErr] = useState(false);
  const [closeupErr, setCloseupErr] = useState(false);
  const [closeupLightboxOpen, setCloseupLightboxOpen] = useState(false);
  const cropUrl = resolveImageUrl(issue.crop_image_url);
  const closeupUrl = resolveImageUrl(issue.closeup_image_url);
  const reviewState = getReviewState(issue);

  return (
    <div className="space-y-3 rounded-[20px] bg-white p-3 ring-1 ring-gray-100">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <p className="text-sm font-extrabold text-gray-950">
          {getCandidateTypeLabel(issue.candidate_type)} {index + 1}
        </p>
        <Badge tone={getIssueStatusTone(issue.status)}>
          {getIssueStatusLabel(issue.status)}
        </Badge>
      </div>

      {/* 상태 + 학생 메모 한 줄 요약 */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl bg-gray-50 px-3 py-2">
        <span className="text-xs text-gray-500">{getCandidateTypeContext(issue.candidate_type).slice(0, 40)}…</span>
        <span className="text-xs font-medium text-blue-700">{reviewState}</span>
        {issue.student_note && (
          <span className="text-xs text-amber-700">메모: {issue.student_note}</span>
        )}
      </div>

      {/* 근접 사진 + crop 좌우 배치 */}
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <p className="text-[10px] font-bold text-gray-500">학생 근접 확인 사진</p>
          {closeupUrl && !closeupErr ? (
            <>
              <button type="button" className="block w-full" onClick={() => setCloseupLightboxOpen(true)}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={closeupUrl}
                  alt="학생 클로즈업 사진"
                  onError={() => setCloseupErr(true)}
                  className="aspect-square w-full rounded-xl bg-gray-100 object-cover"
                />
              </button>
              <ImageLightbox
                open={closeupLightboxOpen}
                src={closeupUrl}
                alt="학생 클로즈업 사진"
                onClose={() => setCloseupLightboxOpen(false)}
              />
            </>
          ) : (
            <div className="flex aspect-square w-full items-center justify-center rounded-xl bg-gray-100 text-[10px] text-gray-400">
              미제출
            </div>
          )}
        </div>
        <div className="space-y-1">
          <p className="text-[10px] font-bold text-gray-500">AI 감지 영역 crop</p>
          {cropUrl && !cropErr ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={cropUrl}
              alt="AI 감지 영역 crop"
              onError={() => setCropErr(true)}
              className="aspect-square w-full rounded-xl bg-gray-100 object-cover"
            />
          ) : (
            <div className="flex aspect-square w-full items-center justify-center rounded-xl bg-gray-100 text-[10px] text-gray-300">
              없음
            </div>
          )}
        </div>
      </div>

      {/* VLM AI 의견 요약 */}
      <div className="rounded-xl bg-blue-50 px-3 py-2 ring-1 ring-blue-100">
        <p className="mb-1 text-[10px] font-bold text-blue-800">AI 참고 의견</p>
        {issue.vlm_reason ? (
          <VlmSummary reason={issue.vlm_reason} />
        ) : (
          <p className="text-xs text-blue-400">근접 촬영 제출 후 표시됩니다.</p>
        )}
      </div>
    </div>
  );
}

function IssueReview({
  inspection,
}: {
  inspection: Inspection;
}) {
  const issues = useMemo(() => inspection.issues ?? [], [inspection.issues]);
  const [selectedIssueId, setSelectedIssueId] = useState<number | null>(issues[0]?.id ?? null);

  useEffect(() => {
    if (issues.length === 0) {
      setSelectedIssueId(null);
      return;
    }
    if (!issues.some((issue) => issue.id === selectedIssueId)) {
      setSelectedIssueId(issues[0].id);
    }
  }, [issues, selectedIssueId]);

  if (issues.length === 0) {
    return (
      <div>
        <p className="mb-2 text-xs font-semibold text-gray-700">AI가 감지한 전후 차이 후보 (0건)</p>
        <p className="rounded-2xl bg-gray-50 py-5 text-center text-xs text-gray-400">
          특별한 차이가 감지되지 않았어요.
        </p>
      </div>
    );
  }

  const selectedIssue = issues.find((issue) => issue.id === selectedIssueId) ?? issues[0];
  const selectedIndex = issues.findIndex((issue) => issue.id === selectedIssue.id);

  return (
    <div className="space-y-3">
      <IssueMap
        finalImageUrl={inspection.final_image_url}
        issues={issues}
        selectedIssueId={selectedIssue.id}
        onSelect={setSelectedIssueId}
      />

      <IssueDetail issue={selectedIssue} index={selectedIndex} />
    </div>
  );
}

export function InspectionDetail({
  inspection,
  room,
  onApprove,
  onReject,
  actionLoading,
  actionError,
}: InspectionDetailProps) {
  const isPending = inspection.status === "pending_review";

  return (
    <div className="space-y-4 pb-3">
      <div className="sticky top-0 z-10 -mx-2 flex items-start justify-between gap-2 rounded-b-[20px] bg-white/95 px-2 py-2 backdrop-blur">
        <div>
          <p className="text-2xl font-extrabold text-gray-950">
            {room ? `${room.room_number}호` : `호실 #${inspection.room_id}`}
          </p>
          {room?.student && (
            <p className="text-sm text-gray-500 mt-0.5">
              {room.student.name} · {room.student.student_number}
            </p>
          )}
        </div>
        <Badge tone={getStatusTone(inspection.status)}>
          {getStatusLabel(inspection.status)}
        </Badge>
      </div>

      {/* 날짜 정보 */}
      <div className="text-xs text-gray-400 space-y-0.5">
        {inspection.submitted_at && (
          <p>제출: {new Date(inspection.submitted_at).toLocaleString("ko-KR")}</p>
        )}
        {inspection.reviewed_at && (
          <p>검토: {new Date(inspection.reviewed_at).toLocaleString("ko-KR")}</p>
        )}
      </div>

      {/* 관리자 피드백 (이미 있는 경우) */}
      {inspection.admin_feedback && (
        <div className="rounded-2xl bg-amber-50 px-3 py-3 ring-1 ring-amber-100">
          <p className="mb-1 text-xs font-bold text-amber-700">관리자 피드백</p>
          <p className="text-xs text-amber-800">{inspection.admin_feedback}</p>
        </div>
      )}

      <div className="rounded-[20px] bg-gray-50 p-3">
        <p className="mb-2 text-xs font-bold text-gray-700">입사 사진 / 퇴사 사진 비교</p>
        <div className="grid grid-cols-2 gap-2">
          <InspectionImage url={inspection.initial_image_url} label="입사 초기 사진" />
          <InspectionImage url={inspection.final_image_url} label="퇴사 최종 사진" />
        </div>
      </div>

      <IssueReview inspection={inspection} />

      {isPending && (
        <div className="space-y-3 rounded-[20px] bg-gray-50 p-3">
          <div>
            <p className="text-xs font-bold text-blue-700">관리자 최종 판단</p>
            <p className="mt-0.5 text-xs text-gray-500">
              전후 사진, 학생의 확인 자료, AI 참고 의견을 검토한 후 판단해 주세요.
            </p>
          </div>
          {actionError && (
            <div className="rounded-2xl bg-red-50 px-3 py-2 text-xs text-red-700 ring-1 ring-red-100">
              {actionError}
            </div>
          )}
          <div className="flex gap-2">
            <Button
              variant="primary"
              size="md"
              fullWidth
              disabled={actionLoading}
              onClick={onApprove}
            >
              {actionLoading ? "처리 중…" : "승인"}
            </Button>
            <Button
              variant="danger"
              size="md"
              fullWidth
              disabled={actionLoading}
              onClick={onReject}
            >
              반려
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
