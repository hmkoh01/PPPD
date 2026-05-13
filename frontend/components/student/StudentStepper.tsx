/**
 * 학생 점검 단계 표시 컴포넌트
 */
import type { RoomStatus } from "@/lib/types";

const STEPS = [
  { label: "인증",     short: "1" },
  { label: "입사 사진", short: "2" },
  { label: "퇴사 사진", short: "3" },
  { label: "차이 확인", short: "4" },
  { label: "관리자 검토", short: "5" },
];

/** RoomStatus → StudentStepper currentStep (1-based 활성 단계) */
export function statusToStep(status: RoomStatus | string): number {
  switch (status) {
    case "ready":          return 1;
    case "checked_in":     return 2;
    case "pending_review": return 4;
    case "approved":       return 5;
    case "rejected":       return 4;
    default:               return 1;
  }
}

interface StudentStepperProps {
  /** 1-based 현재 단계 (1=입사사진, 2=퇴사사진, 3=이슈, 4=검토대기, 5=완료) */
  currentStep: number;
}

export function StudentStepper({ currentStep }: StudentStepperProps) {
  return (
    <div className="flex items-center overflow-x-auto pb-1 -mx-1 px-1">
      {STEPS.map((step, idx) => {
        const stepNum = idx + 1;
        const done   = stepNum < currentStep;
        const active = stepNum === currentStep;
        return (
          <div key={idx} className="flex items-center shrink-0">
            {/* 연결선 */}
            {idx > 0 && (
              <div
                className={`w-5 sm:w-8 h-0.5 mx-0.5 ${
                  done ? "bg-indigo-500" : "bg-gray-200"
                }`}
              />
            )}
            <div className="flex flex-col items-center gap-0.5">
              <div
                className={[
                  "w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-colors",
                  done
                    ? "bg-indigo-600 text-white"
                    : active
                    ? "bg-indigo-100 text-indigo-700 ring-2 ring-indigo-500"
                    : "bg-gray-100 text-gray-400",
                ].join(" ")}
              >
                {done ? "✓" : stepNum}
              </div>
              {/* 레이블: 모바일에서는 active만 표시 */}
              <span
                className={[
                  "text-xs whitespace-nowrap leading-tight",
                  active
                    ? "text-indigo-700 font-semibold"
                    : done
                    ? "text-indigo-400 hidden sm:block"
                    : "text-gray-300 hidden sm:block",
                ].join(" ")}
              >
                {step.label}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
