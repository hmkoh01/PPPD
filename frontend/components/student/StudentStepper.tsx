import type { RoomStatus } from "@/lib/types";

const STEPS = [
  { label: "인증", short: "1" },
  { label: "입사 사진", short: "2" },
  { label: "퇴사 사진", short: "3" },
  { label: "클로즈업", short: "4" },
  { label: "결과", short: "5" },
];

export function statusToStep(status: RoomStatus | string): number {
  switch (status) {
    case "ready":
      return 2;
    case "checked_in":
      return 3;
    case "pending_review":
    case "approved":
    case "rejected":
      return 5;
    default:
      return 1;
  }
}

interface StudentStepperProps {
  currentStep: number;
}

export function StudentStepper({ currentStep }: StudentStepperProps) {
  return (
    <div className="-mx-1 flex items-center overflow-x-auto overflow-y-visible px-1 pb-1 pt-1.5">
      {STEPS.map((step, idx) => {
        const stepNum = idx + 1;
        const done = stepNum < currentStep;
        const active = stepNum === currentStep;
        return (
          <div key={step.label} className="flex shrink-0 items-center">
            {idx > 0 && (
              <div className={`mx-1 h-0.5 w-6 sm:w-9 ${done ? "bg-blue-400" : "bg-gray-200"}`} />
            )}
            <div className="flex flex-col items-center gap-1">
              <div
                className={[
                  "flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors",
                  done
                    ? "bg-blue-600 text-white"
                    : active
                    ? "bg-blue-50 text-blue-700 ring-2 ring-blue-500"
                    : "bg-white text-gray-300 ring-1 ring-gray-200",
                ].join(" ")}
              >
                {done ? "✓" : step.short}
              </div>
              <span
                className={[
                  "whitespace-nowrap text-xs leading-tight",
                  active
                    ? "font-semibold text-blue-700"
                    : done
                    ? "hidden text-blue-400 sm:block"
                    : "hidden text-gray-300 sm:block",
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
