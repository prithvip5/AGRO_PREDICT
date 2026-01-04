import React, { useState } from "react";
import { Input } from "../components/ui/Input";
import { Card } from "../components/ui/Card";

const Weather = () => {
  const [location, setLocation] = useState("Village Center");

  // Static / mock data
  const mockWeather = {
    temperature: "29°C",
    humidity: "68%",
    rainfall: "Light showers",
    summary: "Cloudy with chances of light rain in the afternoon."
  };

  return (
    <div className="pt-3">
      <p className="text-xs text-gray-500 mb-3">
        Check today&apos;s weather before planning irrigation or spraying.
      </p>

      <Input
        label="Your location"
        placeholder="City / village name"
        value={location}
        onChange={(e) => setLocation(e.target.value)}
        helper="This is a mock field. No internet or GPS is required."
      />

      <Card className="mt-2">
        <div className="flex items-center justify-between mb-2">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">
              Location
            </p>
            <p className="text-sm font-semibold text-agro-dark">
              {location || "Not set"}
            </p>
          </div>
          <div className="text-3xl">🌤️</div>
        </div>
        <p className="text-xs text-gray-500">{mockWeather.summary}</p>
      </Card>

      <div className="mt-3 grid gap-3">
        <Card className="flex items-center justify-between">
          <div>
            <p className="text-xs text-gray-500 uppercase">Temperature</p>
            <p className="text-xl font-semibold text-agro-dark">
              {mockWeather.temperature}
            </p>
            <p className="text-[11px] text-gray-500">
              Good for most crops, avoid spraying at peak noon heat.
            </p>
          </div>
          <span className="text-3xl">🌡️</span>
        </Card>

        <Card className="flex items-center justify-between">
          <div>
            <p className="text-xs text-gray-500 uppercase">Humidity</p>
            <p className="text-xl font-semibold text-agro-dark">
              {mockWeather.humidity}
            </p>
            <p className="text-[11px] text-gray-500">
              High humidity can increase pest and disease risk.
            </p>
          </div>
          <span className="text-3xl">💧</span>
        </Card>

        <Card className="flex items-center justify-between">
          <div>
            <p className="text-xs text-gray-500 uppercase">Rainfall</p>
            <p className="text-base font-semibold text-agro-dark">
              {mockWeather.rainfall}
            </p>
            <p className="text-[11px] text-gray-500">
              Plan irrigation lightly, natural moisture will support crops.
            </p>
          </div>
          <span className="text-3xl">🌧️</span>
        </Card>
      </div>

      <p className="mt-4 text-[11px] text-gray-500">
        Data shown here is example data only. Connect to a real weather
        service later without changing this screen.
      </p>
    </div>
  );
};

export default Weather;


