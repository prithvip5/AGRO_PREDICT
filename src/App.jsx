import { Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";

import MobileShell from "./components/layout/MobileShell";

import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import Weather from "./pages/Weather";
import Soil from "./pages/Soil";
import Todo from "./pages/Todo";

/* ---------------- Protected Route ---------------- */
const ProtectedRoute = (props) => {
  const { children } = props;
  const { isAuthenticated, isLoadingAuth } = useAuth();

  if (isLoadingAuth) {
    return null;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

/* ---------------- Routes ---------------- */
const AppRoutes = () => {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <MobileShell title="AgroPredict">
            <Landing />
          </MobileShell>
        }
      />

      <Route
        path="/login"
        element={
          <MobileShell title="Farmer Login" showBack>
            <Login />
          </MobileShell>
        }
      />

      <Route
        path="/register"
        element={
          <MobileShell title="Create Account" showBack>
            <Register />
          </MobileShell>
        }
      />

      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <MobileShell title="Farmer Dashboard">
              <Dashboard />
            </MobileShell>
          </ProtectedRoute>
        }
      />

      <Route
        path="/weather"
        element={
          <ProtectedRoute>
            <MobileShell title="Weather Forecast" showBack>
              <Weather />
            </MobileShell>
          </ProtectedRoute>
        }
      />

      <Route
        path="/soil"
        element={
          <ProtectedRoute>
            <MobileShell title="Soil & Crops" showBack>
              <Soil />
            </MobileShell>
          </ProtectedRoute>
        }
      />

      <Route
        path="/todo"
        element={
          <ProtectedRoute>
            <MobileShell title="Farmer Tasks" showBack>
              <Todo />
            </MobileShell>
          </ProtectedRoute>
        }
      />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

/* ---------------- App Root ---------------- */
const App = () => {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
};

export default App;
