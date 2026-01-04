const mongoose = require("mongoose");

const OtpSchema = new mongoose.Schema({
  mobile: String,
  otp: String,
  createdAt: { type: Date, default: Date.now, expires: 300 }
});

module.exports = mongoose.model("Otp", OtpSchema);
