const Soil = require("../models/SoilRecord");
const CropSel = require("../models/CropSelection");
const History = require("../models/PredictionHistory");
const ml = require("../services/ml.service");

exports.predict = async (req, res) => {
  const farmerId = req.user.id;

  // 1. Save soil data
  const soil = await Soil.create({ farmerId, ...req.body.soil });

  // 2. Weather prediction (10 days)
  const weather = await ml.getWeather10Days(req.body.location);

  // 3. Crop decision
  let result;
  if (req.body.selectedCrop) {
    result = await ml.getCropAdvisory({
      crop: req.body.selectedCrop,
      soil: soil,
      weather: weather
    });
  } else {
    result = await ml.getCropRecommendation({
      soil: soil,
      weather: weather
    });
  }

  // 4. Save history
  await History.create({
    farmerId,
    soil,
    weatherForecast: weather,
    recommendedCrop: result.crop,
    advisory: result.message
  });

  res.json({ weather, result });
};
