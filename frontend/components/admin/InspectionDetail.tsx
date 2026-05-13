"use client";

import { useState } from "react";
import type { Inspection, Issue, Room } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { getStatusLabel, getStatusTone, getIssueStatusLabel, getIssueStatusTone } from "@/lib/status";

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
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-gray-500">{label}</p>
      {url && !err ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt={label}
          onError={() => setErr(true)}
          className="w-full rounded-xl object-cover aspect-video bg-gray-100"
        />
      ) : (
        <div className="w-full rounded-xl aspect-video bg-gray-100 flex items-center justify-center text-gray-400 text-sm">
          사진 없음
        </div>
      )}
    </div>
  );
}

function IssueCard({ issue }: { issue: Issue }) {
  const [cropErr, setCropErr] = useState(false);
  const [closeupErr, setCloseupErr] = useState(false);

  return (
    <div className="border border-gray-100 rounded-xl p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-700">
          이슈 #{issue.id}
        </span>
        <Badge tone={getIssueStatusTone(issue.status)}>
          {getIssueStatusLabel(issue.status)}
        </Badge>
      </div>

      <p className="text-xs text-gray-400">
        위치: ({issue.x}, {issue.y}) / 크기: {issue.width}×{issue.height}
      </p>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <p className="text-xs text-gray-400">크롭 (CV 감지)</p>
          {issue.crop_image_url && !cropErr ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={issue.crop_image_url}
              alt="크롭"
              onError={() => setCropErr(true)}
              className="w-full rounded-lg object-cover aspect-square bg-gray-100"
            />
          ) : (
            <div className="w-full rounded-lg aspect-square bg-gray-100 flex items-center justify-center text-gray-300 text-xs">
              없음
            </div>
          )}
        </div>
        <div className="space-y-1">
          <p className="text-xs text-gray-400">클로즈업 (학생 촬영)</p>
          {issue.closeup_image_url && !closeupErr ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={issue.closeup_image_url}
              alt="클로즈업"
              onError={() => setCloseupErr(true)}
              className="w-full rounded-lg object-cover aspect-square bg-gray-100"
            />
          ) : (
            <div className="w-full rounded-lg aspect-square bg-gray-100 flex items-center justify-center text-gray-300 text-xs">
              없음
            </div>
          )}
        </div>
      </div>

      {issue.vlm_reason && (
        <p className="text-xs text-gray-600 bg-gray-50 rounded-lg px-2 py-1.5 leading-relaxed">
          AI: {issue.vlm_reason}
        </p>
      )}
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
    <div className="space-y-4">
      {/* 헤더 정보 */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-base font-bold text-gray-900">
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
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl px-3 py-2">
          <p className="text-xs font-medium text-yellow-700 mb-0.5">관리자 피드백</p>
          <p className="text-xs text-yellow-800">{inspection.admin_feedback}</p>
        </div>
      )}

      {/* 승인/반려 버튼 */}
      {isPending && (
        <>
          {actionError && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-3 py-2 text-xs text-red-700">
              {actionError}
            </div>
          )}
          <div className="flex gap-2">
            <Button
              variant="primary"
              size="sm"
              fullWidth
              disabled={actionLoading}
              onClick={onApprove}
            >
              {actionLoading ? "처리 중…" : "승인"}
            </Button>
            <Button
              variant="danger"
              size="sm"
              fullWidth
              disabled={actionLoading}
              onClick={onReject}
            >
              반려
            </Button>
          </div>
        </>
      )}

      {/* 이미지 섹션 */}
      <InspectionImage url={inspection.initial_image_url} label="입사 초기 사진" />
      <InspectionImage url={inspection.final_image_url} label="퇴사 최종 사진" />

      {/* 이슈 목록 */}
      <div>
        <p className="text-xs font-semibold text-gray-700 mb-2">
          감지된 차이점 ({inspection.issues?.length ?? 0}건)
        </p>
        {!inspection.issues || inspection.issues.length === 0 ? (
          <p className="text-xs text-gray-400 text-center py-4">
            감지된 차이점 없음
          </p>
        ) : (
          <div className="space-y-2">
            {inspection.issues.map((issue) => (
              <IssueCard key={issue.id} issue={issue} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
