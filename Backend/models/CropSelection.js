const mongoose = require("mongoose");

const CropSelectionSchema = new mongoose.Schema({
  farmerId: mongoose.Schema.Types.ObjectId,
  cropName: String,
  stage: String,
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("CropSelection", CropSelectionSchema);
