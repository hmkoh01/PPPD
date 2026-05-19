"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/Button";

type IntroPhase = "logo" | "typing" | "settle" | "ready";

const COPY = {
  logoAlt: "\uccb4\ud06c\uba54\uc774\ud2b8",
  headlineAi: "\uc804\ud6c4 \ucc28\uc774\ub294 AI\uac00 \ucc3e\uace0,",
  headlineHuman: "\ucd5c\uc885 \ud655\uc778\uc740 \uc0ac\ub78c\uc774 \ud569\ub2c8\ub2e4.",
  studentCta: "\ud559\uc0dd \uc810\uac80 \uc2dc\uc791",
  adminCta: "\uad00\ub9ac\uc790 \ud654\uba74",
  helper: "\ubaa8\ubc14\uc77c \uce74\uba54\ub77c\ub85c \ubc14\ub85c \uc9c4\ud589\ud560 \uc218 \uc788\uc5b4\uc694.",
};

const TYPE_SPEED_MS = 38;

export default function LandingPage() {
  const [phase, setPhase] = useState<IntroPhase>("logo");
  const [logoVisible, setLogoVisible] = useState(false);
  const [typedLine1, setTypedLine1] = useState("");
  const [typedLine2, setTypedLine2] = useState("");

  useEffect(() => {
    const timers: Array<ReturnType<typeof setTimeout>> = [];
    const intervals: Array<ReturnType<typeof setInterval>> = [];

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (prefersReducedMotion) {
      setLogoVisible(true);
      setTypedLine1(COPY.headlineAi);
      setTypedLine2(COPY.headlineHuman);
      setPhase("ready");
      return undefined;
    }

    timers.push(
      setTimeout(() => {
        setLogoVisible(true);
      }, 50),
    );

    const typeLine = (
      text: string,
      setText: (value: string) => void,
      onComplete: () => void,
    ) => {
      let index = 0;
      const interval = setInterval(() => {
        index += 1;
        setText(text.slice(0, index));

        if (index >= text.length) {
          clearInterval(interval);
          onComplete();
        }
      }, TYPE_SPEED_MS);

      intervals.push(interval);
    };

    timers.push(
      setTimeout(() => {
        setPhase("typing");
        typeLine(COPY.headlineAi, setTypedLine1, () => {
          timers.push(
            setTimeout(() => {
              typeLine(COPY.headlineHuman, setTypedLine2, () => {
                timers.push(
                  setTimeout(() => {
                    setPhase("settle");
                    timers.push(
                      setTimeout(() => {
                        setPhase("ready");
                      }, 450),
                    );
                  }, 160),
                );
              });
            }, 80),
          );
        });
      }, 500),
    );

    return () => {
      timers.forEach((timer) => clearTimeout(timer));
      intervals.forEach((interval) => clearInterval(interval));
    };
  }, []);

  const isSettled = phase === "settle" || phase === "ready";
  const isReady = phase === "ready";

  return (
    <div className="relative flex min-h-[100dvh] flex-col overflow-hidden bg-[#F7F8FA] px-6 pt-8 pb-[calc(env(safe-area-inset-bottom)+24px)]">
      <img
        src="/logo.svg"
        alt={COPY.logoAlt}
        className={[
          "absolute z-10 w-auto will-change-transform transition-[left,top,height,opacity,transform] duration-1000 ease-[cubic-bezier(0.22,1,0.36,1)]",
          isSettled
            ? "left-[calc(100%-1.5rem)] top-[calc(env(safe-area-inset-top)+24px)] h-14 -translate-x-full translate-y-0 opacity-100"
            : [
                "left-1/2 top-[24dvh] h-44 -translate-x-1/2 translate-y-0",
                logoVisible ? "scale-100 opacity-100" : "scale-95 opacity-0",
              ].join(" "),
        ].join(" ")}
      />

      <main
        className={[
          "flex flex-1 flex-col items-center justify-center transition-transform duration-700 ease-out",
          isSettled ? "-translate-y-20" : "translate-y-0",
        ].join(" ")}
      >
        <section
          className={[
            "flex w-full flex-col items-center transition-all duration-700 ease-out",
            isSettled ? "-translate-y-6" : "translate-y-0",
          ].join(" ")}
        >
          <div className="h-44 w-full" aria-hidden="true" />

          <div
            className={[
              "min-h-[76px] w-full space-y-1 text-center transition-all duration-700 ease-out",
              isSettled ? "mt-10" : "mt-14",
              phase === "logo" ? "opacity-0" : "opacity-100",
            ].join(" ")}
          >
            <p className="text-2xl font-bold text-gray-800 leading-snug tracking-widest">
              {typedLine1}
              {phase === "typing" && typedLine1.length < COPY.headlineAi.length
                ? "|"
                : ""}
            </p>
            <p className="text-2xl font-bold text-blue-600 leading-snug tracking-widest">
              {typedLine2}
              {phase === "typing" &&
              typedLine1.length === COPY.headlineAi.length &&
              typedLine2.length < COPY.headlineHuman.length
                ? "|"
                : ""}
            </p>
          </div>
        </section>

        <div
          className={[
            "mt-12 w-full space-y-3 transition-all delay-300 duration-700 ease-out",
            isReady
              ? "translate-y-0 opacity-100 pointer-events-auto"
              : "translate-y-5 opacity-0 pointer-events-none",
          ].join(" ")}
        >
          <Link href="/student" className="block">
            <Button variant="primary" size="lg" fullWidth>
              {COPY.studentCta}
            </Button>
          </Link>
          <Link href="/admin" className="block">
            <Button
              variant="secondary"
              size="lg"
              fullWidth
              className="bg-white text-gray-900 ring-1 ring-gray-200 shadow-sm hover:bg-gray-50 active:bg-gray-100"
            >
              {COPY.adminCta}
            </Button>
          </Link>
        </div>
      </main>

      <p
        className={[
          "text-center text-xs text-gray-400 transition-all delay-500 duration-700 ease-out",
          isReady ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
        ].join(" ")}
      >
        {COPY.helper}
      </p>
    </div>
  );
}
