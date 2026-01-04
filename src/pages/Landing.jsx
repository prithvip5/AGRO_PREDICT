import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/Button";

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-full justify-center">
      <div className="text-center mb-10">
        <div className="mx-auto mb-5 inline-flex items-center justify-center rounded-full bg-agro-greenSoft text-agro-green w-16 h-16 text-3xl shadow-soft">
          🌾
        </div>
        <h2 className="text-2xl font-semibold text-agro-dark mb-2">
          Welcome to AgroPredict
        </h2>
        <p className="text-sm text-gray-600 max-w-xs mx-auto">
          Smart crop suggestions using{" "}
          <span className="font-medium">weather</span> and{" "}
          <span className="font-medium">soil</span> data, designed for farmers.
        </p>
      </div>

      <div className="space-y-3">
        <Button onClick={() => navigate("/login")}>Login</Button>
        <Button variant="secondary" onClick={() => navigate("/register")}>
          Register
        </Button>
      </div>

      <p className="mt-8 text-[11px] text-center text-gray-500 px-4">
        Use this app in the field to quickly check today&apos;s weather, enter
        soil values, and get a simple crop recommendation.
      </p>
    </div>
  );
};

export default Landing;


