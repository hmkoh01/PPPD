import type { HTMLAttributes } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
}

export function Card({ title, description, className = "", children, ...rest }: CardProps) {
  return (
    <div
      className={`bg-white rounded-[24px] p-6 shadow-none ring-1 ring-gray-100/80 ${className}`}
      {...rest}
    >
      {(title || description) && (
        <div className="mb-4">
          {title && <h3 className="text-lg font-bold text-gray-950">{title}</h3>}
          {description && <p className="mt-1 text-sm leading-relaxed text-gray-500">{description}</p>}
        </div>
      )}
      {children}
    </div>
  );
}
