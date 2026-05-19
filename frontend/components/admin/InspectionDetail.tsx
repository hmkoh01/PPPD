"use client";

import { useEffect, useMemo, useState } from "react";
import type { Inspection, Issue, Room } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
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
  const imageUrl = resolveImageUrl(url);
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-gray-500">{label}</p>
      {imageUrl && !err ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageUrl}
          alt={label}
          onError={() => setErr(true)}
          className="aspect-video w-full rounded-[18px] bg-gray-100 object-cover"
        />
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
  if (!issue.closeup_image_url) return "클로즈업 미촬영";
  if (!issue.vlm_reason) return "분석 중";
  return "분석 완료";
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
        <p className="text-xs font-bold text-gray-700">전체 퇴사 사진 의심 영역</p>
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
          <p className="text-xs text-gray-400">좌표 정보가 없어 사진 위에 의심 영역을 표시할 수 없습니다.</p>
        </div>
      )}
    </div>
  );
}

function IssueDetail({ issue, index }: { issue: Issue; index: number }) {
  const [cropErr, setCropErr] = useState(false);
  const [closeupErr, setCloseupErr] = useState(false);
  const cropUrl = resolveImageUrl(issue.crop_image_url);
  const closeupUrl = resolveImageUrl(issue.closeup_image_url);
  const reviewState = getReviewState(issue);

  return (
    <div className="space-y-4 rounded-[20px] bg-white p-4 ring-1 ring-gray-100">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-lg font-extrabold text-gray-950">의심 영역 {index + 1}</p>
          <p className="mt-0.5 text-xs text-gray-400">
            위치 ({issue.x}, {issue.y}) · {issue.width}×{issue.height}
          </p>
        </div>
        <Badge tone={getIssueStatusTone(issue.status)}>
          {getIssueStatusLabel(issue.status)}
        </Badge>
      </div>

      <div className="rounded-2xl bg-gray-50 px-3 py-3">
        <p className="text-xs font-bold text-gray-500">촬영 상태</p>
        <p className="mt-1 text-sm font-bold text-gray-900">
          {issue.closeup_image_url ? "클로즈업 촬영 완료" : "클로즈업 미촬영"}
        </p>
        <p className="mt-1 text-xs font-medium text-blue-700">{reviewState}</p>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-bold text-gray-700">학생이 촬영한 클로즈업 사진</p>
        {closeupUrl && !closeupErr ? (
            // eslint-disable-next-line @next/next/no-img-element
          <img
            src={closeupUrl}
            alt="학생이 촬영한 클로즈업 사진"
            onError={() => setCloseupErr(true)}
            className="aspect-video w-full rounded-2xl bg-gray-100 object-cover"
          />
        ) : (
          <div className="flex aspect-video w-full items-center justify-center rounded-2xl bg-gray-100 text-xs text-gray-400">
            아직 클로즈업 사진이 없습니다.
          </div>
        )}
      </div>

      <div className="rounded-2xl bg-blue-50 px-4 py-3 ring-1 ring-blue-100">
        <p className="text-xs font-bold text-blue-800">AI 확인 결과</p>
        <p className="mt-2 text-sm leading-relaxed text-blue-950">
          {issue.vlm_reason || "클로즈업 촬영 후 AI 분석 결과가 표시됩니다."}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-blue-700">
          AI 확인 결과는 참고용이며, 최종 판단은 관리자가 검토합니다.
        </p>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-bold text-gray-500">보조 정보: 감지 영역 crop</p>
        {cropUrl && !cropErr ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={cropUrl}
            alt="감지 영역 crop"
            onError={() => setCropErr(true)}
            className="aspect-video w-full rounded-2xl bg-gray-100 object-cover"
          />
        ) : (
          <div className="flex aspect-video w-full items-center justify-center rounded-2xl bg-gray-100 text-xs text-gray-300">
            crop 없음
          </div>
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
        <p className="mb-2 text-xs font-semibold text-gray-700">감지된 차이점 (0건)</p>
        <p className="rounded-2xl bg-gray-50 py-5 text-center text-xs text-gray-400">
          감지된 차이점 없음
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

      <div className="space-y-3 rounded-[20px] bg-gray-50 p-3">
        <p className="text-xs font-bold text-gray-700">입사 사진 / 퇴사 사진 비교</p>
        <InspectionImage url={inspection.initial_image_url} label="입사 초기 사진" />
        <InspectionImage url={inspection.final_image_url} label="퇴사 최종 사진" />
      </div>

      <IssueReview inspection={inspection} />

      {isPending && (
        <div className="space-y-3 rounded-[20px] bg-gray-50 p-3">
          <p className="text-xs font-bold text-blue-700">처리 필요</p>
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
