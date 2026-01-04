const Otp = require("../models/Otp");
const Farmer = require("../models/Farmer");
const otpGenerator = require("otp-generator");
const jwt = require("jsonwebtoken");

exports.sendOtp = async (req, res) => {
  const { mobile } = req.body;
  const otp = otpGenerator.generate(6, { digits: true });
  await Otp.create({ mobile, otp });
  console.log("OTP:", otp); // replace with SMS later
  res.json({ message: "OTP sent" });
};

exports.verifyOtp = async (req, res) => {
  const { mobile, otp } = req.body;
  const record = await Otp.findOne({ mobile, otp });
  if (!record) return res.status(400).json({ error: "Invalid OTP" });

  let farmer = await Farmer.findOne({ mobile });
  if (!farmer) farmer = await Farmer.create({ mobile });

  const token = jwt.sign({ id: farmer._id }, process.env.JWT_SECRET);
  res.json({ token, farmer });
};
