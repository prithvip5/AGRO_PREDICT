const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/auth", require("./routes/auth.routes"));
app.use("/api/farmer", require("./routes/farmer.routes"));
app.use("/api/predict", require("./routes/prediction.routes"));

module.exports = app;
