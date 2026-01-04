import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";

const Register = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [phoneNumber, setPhoneNumber] = useState("");
  const [otp, setOtp] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [isSendingOtp, setIsSendingOtp] = useState(false);

  const handleSendOtp = async (e) => {
    e.preventDefault();
    if (!phoneNumber || phoneNumber.length < 10) {
      return;
    }
    
    setIsSendingOtp(true);
    // Mock OTP sending - in real app, this would call an API
    setTimeout(() => {
      setIsSendingOtp(false);
      setOtpSent(true);
      // For demo purposes, set a mock OTP (in production, this comes from backend)
      // You can use any 6-digit number to verify: e.g., "123456"
    }, 1000);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const otpValue = otp.trim();
    if (!otpValue || otpValue.length !== 6) {
      alert("Please enter a valid 6-digit OTP");
      return;
    }
    // Mock OTP verification - in real app, verify with backend
    // For demo, accept any 6-digit OTP
    console.log("Verifying OTP:", otpValue);
    login(phoneNumber);
    navigate("/dashboard");
  };

  const handleVerifyClick = () => {
    const otpValue = otp.trim();
    if (!otpValue || otpValue.length !== 6) {
      alert("Please enter a valid 6-digit OTP");
      return;
    }
    // Mock OTP verification - in real app, verify with backend
    // For demo, accept any 6-digit OTP
    console.log("Verifying OTP:", otpValue);
    login(phoneNumber);
    navigate("/dashboard");
  };

  return (
    <div className="pt-4">
      <h2 className="text-xl font-semibold text-agro-dark mb-1">
        Create your farm account
      </h2>
      <p className="text-xs text-gray-500 mb-5">
        Keep your fields, tasks and soil readings in one simple place.
      </p>

      {!otpSent ? (
        <form onSubmit={handleSendOtp} className="space-y-2">
          <Input
            label="Phone Number"
            type="tel"
            placeholder="+91 9876543210"
            value={phoneNumber}
            onChange={(e) => setPhoneNumber(e.target.value)}
            helper="We'll send you a verification code"
          />

          <div className="pt-2">
            <Button 
              type="submit" 
              disabled={!phoneNumber || phoneNumber.length < 10 || isSendingOtp}
            >
              {isSendingOtp ? "Sending OTP..." : "Send OTP"}
            </Button>
          </div>
        </form>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-2">
          <div className="mb-2">
            <p className="text-xs text-gray-600 mb-2">
              OTP sent to <span className="font-medium">{phoneNumber}</span>
            </p>
            <button
              type="button"
              onClick={() => {
                setOtpSent(false);
                setOtp("");
              }}
              className="text-xs text-agro-green font-medium"
            >
              Change number
            </button>
          </div>

          <Input
            label="Enter OTP"
            type="text"
            placeholder="123456"
            value={otp}
            onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))}
            helper="Enter the 6-digit code sent to your phone (Demo: any 6 digits work)"
            maxLength={6}
            inputMode="numeric"
          />

          <div className="pt-2">
            <Button 
              type="button"
              onClick={handleVerifyClick}
              disabled={!otp || otp.trim().length !== 6}
            >
              Verify & Register
            </Button>
          </div>
          
          {otp && otp.length !== 6 && (
            <p className="text-xs text-gray-500 mt-1 text-center">
              Enter 6 digits to continue
            </p>
          )}
          
          <p className="text-xs text-gray-400 mt-2 text-center italic">
            💡 Demo mode: Any 6-digit number will work
          </p>

          <div className="text-center mt-3">
            <button
              type="button"
              onClick={handleSendOtp}
              className="text-xs text-agro-green font-medium"
            >
              Resend OTP
            </button>
          </div>
        </form>
      )}

      <p className="mt-4 text-xs text-gray-600">
        Already have an account?{" "}
        <Link to="/login" className="text-agro-green font-medium">
          Login here
        </Link>
      </p>
    </div>
  );
};

export default Register;


