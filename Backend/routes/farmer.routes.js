// routes/farmer.routes.js
const router = require("express").Router();
const Farmer = require("../models/Farmer");

// Save / update farmer location
router.post("/location", async (req, res) => {
  try {
    const { farmerId, lat, lng } = req.body;

    await Farmer.findByIdAndUpdate(farmerId, {
      location: { lat, lng }
    });

    res.json({ message: "Farmer location saved successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
