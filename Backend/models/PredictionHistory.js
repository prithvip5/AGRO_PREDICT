const mongoose = require("mongoose");

const PredictionSchema = new mongoose.Schema({
  farmerId: mongoose.Schema.Types.ObjectId,
  soil: Object,
  weatherForecast: Object,
  recommendedCrop: String,
  advisory: String,
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("PredictionHistory", PredictionSchema);
