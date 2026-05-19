"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

const NAV = [
  { href: "/admin", label: "대시보드" },
  { href: "/admin/rooms", label: "호실" },
  { href: "/admin/reviews", label: "리뷰" },
];

interface AdminShellProps {
  children: React.ReactNode;
}

export function AdminShell({ children }: AdminShellProps) {
  const pathname = usePathname();
  const isActive = (href: string) => {
    if (href === "/admin") {
      return pathname === "/admin";
    }

    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <div className="flex min-h-[100dvh] bg-[#F7F8FA]">
      <aside className="hidden w-56 flex-col gap-1 border-r border-gray-100 bg-white px-4 py-7 lg:flex">
        <div className="mb-7 px-3">
          <span className="text-lg font-bold text-gray-950">퇴사 점검</span>
          <p className="mt-1 text-xs font-medium text-gray-400">기숙사 관리자</p>
        </div>
        {NAV.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={[
                "flex items-center gap-2.5 rounded-2xl px-4 py-3 text-sm font-semibold transition-colors",
                active ? "bg-blue-50 text-blue-700" : "text-gray-600 hover:bg-gray-50",
              ].join(" ")}
            >
              {item.label}
            </Link>
          );
        })}
      </aside>

      <nav className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-3 border-t border-gray-200 bg-white px-3 py-3 pb-[calc(env(safe-area-inset-bottom)+0.75rem)] shadow-[0_-10px_30px_rgba(15,23,42,0.08)] lg:hidden">
        {NAV.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={[
                "rounded-2xl py-3 text-center text-sm font-extrabold transition-colors",
                active ? "bg-blue-50 text-blue-700" : "text-gray-500",
              ].join(" ")}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <main className="mx-auto w-full max-w-lg flex-1 px-5 py-4 pb-28 lg:max-w-5xl lg:px-6 lg:py-5 lg:pb-8">
        <div className="mb-3 flex items-center justify-between gap-3 pt-[env(safe-area-inset-top)]">
          <Link
            href="/"
            className="inline-flex h-9 items-center rounded-full bg-white/80 px-3 text-sm font-semibold text-gray-600 ring-1 ring-gray-100 shadow-sm transition-colors active:bg-gray-100"
          >
            처음으로
          </Link>
          <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-bold text-gray-500">
            관리자
          </span>
        </div>
        {children}
      </main>
    </div>
  );
}
