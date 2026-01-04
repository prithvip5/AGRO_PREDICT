import React from "react";

export const Input = ({
  label,
  type = "text",
  helper,
  className = "",
  ...props
}) => {
  return (
    <label className={`flex flex-col gap-1 mb-3 ${className}`}>
      {label && (
        <span className="text-sm font-medium text-gray-800">{label}</span>
      )}
      <input
        type={type}
        className="w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm shadow-sm focus:border-agro-green focus:ring-2 focus:ring-agro-green/40 outline-none"
        {...props}
      />
      {helper && (
        <span className="text-[11px] text-gray-500 leading-tight">
          {helper}
        </span>
      )}
    </label>
  );
};


