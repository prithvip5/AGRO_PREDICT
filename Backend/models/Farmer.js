const mongoose = require("mongoose");

const FarmerSchema = new mongoose.Schema({
  mobile: { type: String, unique: true, required: true },
  language: { type: String, default: "en" },
  location: {
    lat: Number,
    lng: Number
  },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Farmer", FarmerSchema);
