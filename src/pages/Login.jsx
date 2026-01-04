import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";

const Login = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    // Mock auth: accept any email & password
    login(email || "farmer@example.com");
    navigate("/dashboard");
  };

  return (
    <div className="pt-4">
      <h2 className="text-xl font-semibold text-agro-dark mb-1">
        Login to your farm
      </h2>
      <p className="text-xs text-gray-500 mb-5">
        Use your email to access weather, soil and daily task tools.
      </p>

      <form onSubmit={handleSubmit} className="space-y-2">
        <Input
          label="Email"
          type="email"
          placeholder="farmer@village.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Input
          label="Password"
          type="password"
          placeholder="••••••••"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <div className="pt-2">
          <Button type="submit">Login</Button>
        </div>
      </form>

      <p className="mt-4 text-xs text-gray-600">
        New to AgroPredict?{" "}
        <Link to="/register" className="text-agro-green font-medium">
          Create an account
        </Link>
      </p>
    </div>
  );
};

export default Login;


