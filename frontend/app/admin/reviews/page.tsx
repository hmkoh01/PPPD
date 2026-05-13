"use client";

import { useCallback, useEffect, useState } from "react";
import { AdminShell } from "@/components/admin/AdminShell";
import { ReviewList } from "@/components/admin/ReviewList";
import { InspectionDetail } from "@/components/admin/InspectionDetail";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import {
  getAdminRooms,
  getAdminInspections,
  getAdminInspection,
  approveInspection,
  rejectInspection,
} from "@/lib/api";
import type { Inspection, Room, RoomStatus } from "@/lib/types";

const STATUS_FILTERS: { label: string; value: RoomStatus | "" }[] = [
  { label: "검토 대기", value: "pending_review" },
  { label: "승인 완료", value: "approved" },
  { label: "재점검 요청", value: "rejected" },
  { label: "전체", value: "" },
];

export default function AdminReviewsPage() {
  const [roomsById, setRoomsById] = useState<Map<number, Room>>(new Map());
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [filter, setFilter] = useState<RoomStatus | "">("pending_review");
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<Inspection | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // 반려 modal
  const [rejectModalOpen, setRejectModalOpen] = useState(false);
  const [rejectFeedback, setRejectFeedback] = useState("");

  // 승인/반려 action state
  const [actionLoading, setActionLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  // 목록 로드 (rooms + inspections 병렬)
  const loadList = useCallback(
    (statusFilter: RoomStatus | "") => {
      setListLoading(true);
      setListError(null);
      Promise.all([
        getAdminRooms(),
        getAdminInspections(statusFilter || undefined),
      ])
        .then(([rooms, insp]) => {
          const map = new Map<number, Room>();
          rooms.forEach((r) => map.set(r.id, r));
          setRoomsById(map);
          setInspections(insp);
        })
        .catch((err) =>
          setListError(err instanceof Error ? err.message : "불러오기 실패")
        )
        .finally(() => setListLoading(false));
    },
    []
  );

  useEffect(() => {
    loadList(filter);
  }, [filter, loadList]);

  // 상세 로드
  const loadDetail = useCallback((id: number) => {
    setSelectedId(id);
    setDetail(null);
    setDetailLoading(true);
    setActionError(null);
    setActionSuccess(null);
    getAdminInspection(id)
      .then(setDetail)
      .catch((err) =>
        setActionError(err instanceof Error ? err.message : "상세 로드 실패")
      )
      .finally(() => setDetailLoading(false));
  }, []);

  // 승인
  const handleApprove = async () => {
    if (!selectedId) return;
    setActionLoading(true);
    setActionError(null);
    setActionSuccess(null);
    try {
      const updated = await approveInspection(selectedId);
      setDetail(updated);
      setActionSuccess("승인이 완료되었습니다.");
      loadList(filter);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "승인 실패");
    } finally {
      setActionLoading(false);
    }
  };

  // 반려 submit
  const handleRejectSubmit = async () => {
    if (!selectedId || !rejectFeedback.trim()) return;
    setActionLoading(true);
    setActionError(null);
    setActionSuccess(null);
    try {
      const updated = await rejectInspection(selectedId, {
        admin_feedback: rejectFeedback.trim(),
      });
      setDetail(updated);
      setRejectModalOpen(false);
      setRejectFeedback("");
      setActionSuccess("반려 처리가 완료되었습니다.");
      loadList(filter);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "반려 실패");
    } finally {
      setActionLoading(false);
    }
  };

  const detailRoom = detail ? (roomsById.get(detail.room_id) ?? null) : null;

  return (
    <AdminShell>
      <div className="space-y-4">
        <h1 className="text-xl font-bold text-gray-900">점검 리뷰</h1>

        {/* 필터 탭 */}
        <div className="flex gap-2 flex-wrap">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => {
                setFilter(f.value);
                setSelectedId(null);
                setDetail(null);
                setActionSuccess(null);
              }}
              className={[
                "px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
                filter === f.value
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200",
              ].join(" ")}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* 에러 */}
        {listError && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            {listError}
            <button
              className="ml-2 underline"
              onClick={() => loadList(filter)}
            >
              다시 시도
            </button>
          </div>
        )}

        {/* 성공 메시지 */}
        {actionSuccess && (
          <div className="bg-green-50 border border-green-200 rounded-xl px-4 py-3 text-sm text-green-700">
            {actionSuccess}
          </div>
        )}

        <div className="flex gap-4 items-start">
          {/* 목록 */}
          <div className="flex-1 min-w-0">
            {listLoading ? (
              <div className="space-y-2">
                {[...Array(4)].map((_, i) => (
                  <div
                    key={i}
                    className="h-16 bg-gray-100 rounded-xl animate-pulse"
                  />
                ))}
              </div>
            ) : (
              <ReviewList
                inspections={inspections}
                roomsById={roomsById}
                onSelect={loadDetail}
                selectedId={selectedId}
              />
            )}
          </div>

          {/* 상세 패널 (md 이상) */}
          {(selectedId || detailLoading) && (
            <div className="hidden md:block w-80 flex-shrink-0 bg-white border border-gray-100 rounded-2xl p-4 max-h-screen overflow-y-auto sticky top-4">
              {detailLoading ? (
                <div className="space-y-3">
                  {[...Array(4)].map((_, i) => (
                    <div
                      key={i}
                      className="h-24 bg-gray-100 rounded-xl animate-pulse"
                    />
                  ))}
                </div>
              ) : detail ? (
                <InspectionDetail
                  inspection={detail}
                  room={detailRoom}
                  onApprove={handleApprove}
                  onReject={() => {
                    setRejectFeedback("");
                    setActionError(null);
                    setRejectModalOpen(true);
                  }}
                  actionLoading={actionLoading}
                  actionError={actionError}
                />
              ) : null}
            </div>
          )}
        </div>

        {/* 모바일: 선택된 점검 상세 (별도 섹션) */}
        {detail && !detailLoading && (
          <div className="md:hidden bg-white border border-gray-100 rounded-2xl p-4">
            <InspectionDetail
              inspection={detail}
              room={detailRoom}
              onApprove={handleApprove}
              onReject={() => {
                setRejectFeedback("");
                setActionError(null);
                setRejectModalOpen(true);
              }}
              actionLoading={actionLoading}
              actionError={actionError}
            />
          </div>
        )}
      </div>

      {/* 반려 Modal */}
      <Modal
        open={rejectModalOpen}
        onClose={() => setRejectModalOpen(false)}
        title="점검 반려"
      >
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            반려 사유를 학생에게 전달할 메시지로 입력해 주세요.
          </p>
          <textarea
            value={rejectFeedback}
            onChange={(e) => setRejectFeedback(e.target.value)}
            placeholder="예) 화장실 청결 상태가 기준에 미달합니다. 재점검 후 제출해 주세요."
            rows={4}
            className="w-full px-3 py-2 rounded-xl border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-red-400 resize-none"
            disabled={actionLoading}
          />
          {actionError && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-3 py-2 text-xs text-red-700">
              {actionError}
            </div>
          )}
          <div className="flex gap-2 justify-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setRejectModalOpen(false)}
              disabled={actionLoading}
            >
              취소
            </Button>
            <Button
              variant="danger"
              size="sm"
              disabled={actionLoading || !rejectFeedback.trim()}
              onClick={handleRejectSubmit}
            >
              {actionLoading ? "처리 중…" : "반려 확정"}
            </Button>
          </div>
        </div>
      </Modal>
    </AdminShell>
  );
}
