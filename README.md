## AgroPredict – Mobile-First Farmer UI (Frontend Only)

AgroPredict is a **React + Vite + Tailwind CSS** mobile-first web app designed for farmers.  
It provides mock flows for:

- **Landing** (welcome + CTA)
- **Login / Register** (mock authentication, no backend)
- **Dashboard** (card-based home)
- **Weather Forecast** (static weather cards)
- **Soil Property Input** (NPK + pH, mock recommendation)
- **Farmer To-Do List** (localStorage-based tasks)

All screens are optimized for **smartphones** with large tap targets, clear typography, and agriculture-themed colors.

---

### 1. Requirements

- Node.js 18+ and npm

---

### 2. Install Dependencies

From the project root (`AgroPredict`):

```bash
npm install
```

---

### 3. Run in Development

```bash
npm run dev
```

Then open the printed URL in your browser (usually `http://localhost:5173`).

Use your mobile phone or dev tools device mode to see the **mobile app-like UI**.

---

### 4. Build for Production

```bash
npm run build
```

To preview the production build locally:

```bash
npm run preview
```

---

### 5. Tech Notes

- **Frontend only** – no real API calls, authentication is mocked in `AuthContext`.
- **Routing** – handled with `react-router-dom`.
- **Styling** – Tailwind CSS with a small custom color palette (`agro.*`).
- **State**:
  - Auth info stored in `localStorage` (mock).
  - Farmer tasks stored in `localStorage`.

You can later plug real APIs or ML models behind the existing UI without changing the UX flow.


