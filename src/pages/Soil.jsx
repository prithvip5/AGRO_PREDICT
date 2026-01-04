import React, { useState } from "react";
import { Input } from "../components/ui/Input";
import { Select } from "../components/ui/Select";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

const Soil = () => {
  const [soilMoisture, setSoilMoisture] = useState("");
  const [temperature, setTemperature] = useState("");
  const [ph, setPh] = useState("");
  const [showResult, setShowResult] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setShowResult(true);
  };

  return (
    <div className="pt-3 pb-6">
      <p className="text-xs text-gray-500 mb-3">
        Enter simple soil readings to get a quick crop suggestion. Values can
        be from lab reports or field kits.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-2 gap-3 mb-1">
          <Select
            label="Soil Moisture"
            value={soilMoisture}
            onChange={(e) => setSoilMoisture(e.target.value)}
          >
            <option value="">Select...</option>
            <option value="wet">Wet</option>
            <option value="dry">Dry</option>
          </Select>
          <Input
            label="Temperature from Sun (°C)"
            type="number"
            placeholder="e.g. 35"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
          />
          <Input
            label="Soil pH"
            type="number"
            step="0.1"
            placeholder="e.g. 6.5"
            value={ph}
            onChange={(e) => setPh(e.target.value)}
          />
        </div>

        <div className="mb-3">
          <Card className="bg-agro-greenSoft/70">
            <p className="text-xs text-gray-600 mb-1">
              Mock weather summary for your field
            </p>
            <p className="text-sm font-medium text-agro-dark">
              Warm day with light showers expected in the evening.
            </p>
            <p className="mt-1 text-[11px] text-gray-500">
              Temperature around 29°C, humidity near 70%. Good for rice and
              other water-loving crops.
            </p>
          </Card>
        </div>

        <Button type="submit">Get Crop Recommendation</Button>
      </form>

      {showResult && (
        <Card className="mt-4 border border-agro-greenSoft bg-white">
          <p className="text-xs text-agro-green font-semibold uppercase tracking-wide mb-1">
            Recommended Crop (Mock)
          </p>
          <h3 className="text-lg font-semibold text-agro-dark mb-1">
            Recommended Crop: Rice
          </h3>
          <p className="text-xs text-gray-600 mb-2">
            Based on the entered soil moisture, temperature, and pH values together with warm, humid
            conditions.
          </p>
          <ul className="text-[11px] text-gray-500 list-disc list-inside space-y-1">
            <li>Ensure proper field leveling for uniform water depth.</li>
            <li>Avoid waterlogging young seedlings for too long.</li>
            <li>Monitor for pests in high humidity conditions.</li>
          </ul>
        </Card>
      )}

      <p className="mt-4 text-[11px] text-gray-500">
        This is a demo recommendation. Connect to a machine learning model or
        expert rules later using these same input fields.
      </p>
    </div>
  );
};

export default Soil;


