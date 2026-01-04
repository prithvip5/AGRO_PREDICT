const Farmer = require("../models/Farmer");

exports.updateLocation = async (req, res) => {
  await Farmer.findByIdAndUpdate(req.user.id, {
    location: req.body
  });
  res.json({ message: "Location saved" });
};
