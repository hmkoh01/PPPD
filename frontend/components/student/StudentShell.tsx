"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

interface StudentShellProps {
  children: React.ReactNode;
  title?: string;
  back?: { label: string; href: string };
  trailing?: React.ReactNode;
}

const STUDENT_NAV = [
  { href: "/student/checkin", label: "\uc785\uc0ac" },
  { href: "/student/checkout", label: "\ud1f4\uc0ac" },
  { href: "/student/issues", label: "\ud074\ub85c\uc988\uc5c5" },
  { href: "/student/result", label: "\uacb0\uacfc" },
];

export function StudentShell({ children, title, back, trailing }: StudentShellProps) {
  const pathname = usePathname();
  const showNav = pathname !== "/student/login";
  const homeLink = back ?? { label: "\ucc98\uc74c\uc73c\ub85c", href: "/" };

  const isActive = (href: string) => {
    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <div className="flex min-h-[100dvh] flex-col bg-[#F7F8FA]">
      <header className="sticky top-0 z-20 border-b border-gray-100/70 bg-[#F7F8FA]/90 backdrop-blur">
        <div className="mx-auto flex min-h-16 max-w-lg items-center gap-3 px-5 pt-[env(safe-area-inset-top)]">
          <Link
            href={homeLink.href}
            className="inline-flex h-9 shrink-0 items-center rounded-full bg-white/80 px-3 text-sm font-semibold text-gray-600 ring-1 ring-gray-100 shadow-sm transition-colors active:bg-gray-100"
          >
            {homeLink.label}
          </Link>
          {title && (
            <span className="min-w-0 flex-1 truncate text-center text-base font-bold text-gray-950">
              {title}
            </span>
          )}
          <span className="flex w-[76px] shrink-0 justify-end">
            {trailing && (
              <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-extrabold text-blue-700 ring-1 ring-blue-100">
                {trailing}
              </span>
            )}
          </span>
        </div>
      </header>

      <main
        className={[
          "mx-auto w-full max-w-lg flex-1 space-y-6 px-5 py-7",
          showNav ? "pb-32" : "pb-8",
        ].join(" ")}
      >
        {children}
      </main>

      {showNav && (
        <nav className="fixed inset-x-0 bottom-0 z-20 border-t border-gray-200 bg-white pb-[env(safe-area-inset-bottom)] shadow-[0_-10px_30px_rgba(15,23,42,0.08)]">
          <div className="mx-auto flex max-w-lg gap-1 overflow-x-auto px-2 py-3 text-xs [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            {STUDENT_NAV.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={[
                    "min-w-[60px] flex-1 rounded-2xl py-3 text-center font-extrabold transition-colors",
                    active
                      ? "bg-blue-50 text-blue-700"
                      : "text-gray-500 hover:bg-gray-50 hover:text-gray-800",
                  ].join(" ")}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </nav>
      )}
    </div>
  );
}
