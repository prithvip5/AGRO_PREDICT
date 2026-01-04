import React from "react";
import { useNavigate } from "react-router-dom";
import { Card } from "../components/ui/Card";
import { useAuth } from "../context/AuthContext";
import { Button } from "../components/ui/Button";

const Dashboard = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const displayName = user?.phoneNumber || "Farmer";

  return (
    <div className="pt-3 pb-4">
      <div className="mb-4">
        <p className="text-xs text-gray-500">Good day,</p>
        <h2 className="text-xl font-semibold text-agro-dark">
          {displayName}
        </h2>
        <p className="mt-1 text-xs text-gray-500">
          Quickly check today&apos;s weather, soil, and your field tasks.
        </p>
      </div>

      <div className="grid gap-3">
        <Card
          onClick={() => navigate("/weather")}
          className="bg-gradient-to-r from-agro-green to-emerald-600 text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-emerald-100">
                Today&apos;s sky
              </p>
              <h3 className="text-lg font-semibold mt-1">Weather Forecast</h3>
              <p className="text-xs mt-1 text-emerald-100">
                See rain chance, temperature and humidity.
              </p>
            </div>
            <div className="text-4xl">🌤️</div>
          </div>
        </Card>

        <Card
          onClick={() => navigate("/soil")}
          className="bg-gradient-to-r from-agro-greenSoft to-amber-50"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-wide text-agro-brown">
                Field readings
              </p>
              <h3 className="text-lg font-semibold text-agro-dark mt-1">
                Soil Properties
              </h3>
              <p className="text-xs mt-1 text-gray-600">
                Enter soil moisture, temperature &amp; pH to get a crop suggestion.
              </p>
            </div>
            <div className="text-4xl">🌱</div>
          </div>
        </Card>

        <Card
          onClick={() => navigate("/todo")}
          className="bg-gradient-to-r from-white to-agro-greenSoft"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-wide text-gray-500">
                Today&apos;s work
              </p>
              <h3 className="text-lg font-semibold text-agro-dark mt-1">
                Farmer To-Do List
              </h3>
              <p className="text-xs mt-1 text-gray-600">
                Plan irrigation, spraying, sowing and more.
              </p>
            </div>
            <div className="text-4xl">📝</div>
          </div>
        </Card>
      </div>

      <div className="mt-6">
        <Button
          variant="ghost"
          className="text-xs font-medium text-gray-500"
          onClick={logout}
        >
          Logout
        </Button>
      </div>
    </div>
  );
};

export default Dashboard;


