const mongoose = require("mongoose");

const SoilRecordSchema = new mongoose.Schema({
  farmerId: mongoose.Schema.Types.ObjectId,
  ph: Number,
  moisture: Number,
  sunlight: Number,
  temperature: Number,
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("SoilRecord", SoilRecordSchema);
