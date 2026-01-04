import React from "react";

export const Button = ({
  children,
  variant = "primary",
  className = "",
  disabled,
  ...props
}) => {
  const base =
    "w-full inline-flex items-center justify-center rounded-full px-4 py-3 text-base font-semibold active:scale-[0.97] transition shadow-soft";

  const variants = {
    primary:
      "bg-agro-green text-white hover:bg-emerald-700 focus:ring-2 focus:ring-offset-2 focus:ring-agro-green/40",
    secondary:
      "bg-white text-agro-green border border-agro-greenSoft hover:bg-agro-greenSoft focus:ring-2 focus:ring-offset-2 focus:ring-agro-green/30",
    ghost:
      "bg-transparent text-gray-600 hover:bg-gray-100 shadow-none border border-transparent"
  };

  const disabledStyles = disabled
    ? "opacity-50 cursor-not-allowed pointer-events-none"
    : "";

  return (
    <button
      className={`${base} ${variants[variant]} ${disabledStyles} ${className}`}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
};


