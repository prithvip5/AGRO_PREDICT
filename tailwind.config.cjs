/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        agro: {
          green: "#1c7c54",
          greenSoft: "#e5f4ea",
          brown: "#a07135",
          sand: "#f5f3ef",
          dark: "#1b1f23"
        }
      },
      fontFamily: {
        sans: ["system-ui", "ui-sans-serif", "sans-serif"]
      },
      boxShadow: {
        soft: "0 10px 25px rgba(0,0,0,0.06)"
      },
      borderRadius: {
        card: "1.25rem"
      }
    }
  },
  plugins: []
};


