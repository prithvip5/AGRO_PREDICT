const axios = require("axios");

exports.getWeather10Days = async (data) => {
  return (await axios.post(`${process.env.PYTHON_API}/weather-10days`, data)).data;
};

exports.getCropRecommendation = async (data) => {
  return (await axios.post(`${process.env.PYTHON_API}/crop-recommend`, data)).data;
};

exports.getCropAdvisory = async (data) => {
  return (await axios.post(`${process.env.PYTHON_API}/crop-advisory`, data)).data;
};
