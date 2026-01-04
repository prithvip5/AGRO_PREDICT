// routes/prediction.routes.js
const router = require("express").Router();
const { predict } = require("../controllers/prediction.controller");

router.post("/", predict);

module.exports = router;
