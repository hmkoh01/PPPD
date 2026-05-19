import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "danger" | "ghost";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  fullWidth?: boolean;
}

const variantClasses: Record<Variant, string> = {
  primary:
    "bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 shadow-soft",
  secondary:
    "bg-gray-100 text-gray-900 hover:bg-gray-200 active:bg-gray-300",
  danger:
    "bg-red-500 text-white hover:bg-red-600 active:bg-red-700 shadow-soft",
  ghost:
    "bg-transparent text-gray-600 hover:bg-gray-100 active:bg-gray-200",
};

const sizeClasses: Record<Size, string> = {
  sm: "h-10 px-3 text-sm rounded-xl",
  md: "h-12 px-4 text-base rounded-2xl",
  lg: "h-14 px-6 text-lg rounded-[20px] font-semibold",
};

export function Button({
  variant = "primary",
  size = "md",
  fullWidth = false,
  className = "",
  disabled,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={[
        "inline-flex items-center justify-center gap-2 font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-50",
        variantClasses[variant],
        sizeClasses[size],
        fullWidth ? "w-full" : "",
        disabled ? "cursor-not-allowed bg-gray-200 text-gray-400 shadow-none pointer-events-none" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      disabled={disabled}
      {...rest}
    >
      {children}
    </button>
  );
}
