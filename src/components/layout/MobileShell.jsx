import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";

const navItems = [
  { label: "Home", path: "/dashboard", icon: "🏠" },
  { label: "Weather", path: "/weather", icon: "🌤️" },
  { label: "Soil", path: "/soil", icon: "🌱" },
  { label: "To-Do", path: "/todo", icon: "📝" },
];

const MobileShell = (props) => {
  const { title, showBack = false, children } = props;

  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();

  const isAuthScreen =
    location.pathname === "/" ||
    location.pathname === "/login" ||
    location.pathname === "/register";

  const handleBack = () => {
    if (location.pathname === "/dashboard") {
      navigate("/");
    } else {
      navigate(-1);
    }
  };

  return (
    <div className="min-h-screen flex justify-center bg-agro-sand">
      <div className="flex flex-col w-full max-w-md min-h-screen bg-gradient-to-b from-white to-agro-sand shadow-soft md:rounded-3xl md:my-4 overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between px-5 pt-4 pb-2">
          <div className="flex items-center gap-2">
            {showBack && (
              <button
                onClick={handleBack}
                className="mr-1 rounded-full bg-agro-greenSoft text-agro-green p-2 text-lg shadow-sm active:scale-95 transition"
                aria-label="Back"
              >
                ←
              </button>
            )}

            <div>
              <h1 className="text-lg font-semibold text-agro-dark leading-tight">
                {title || "AgroPredict"}
              </h1>

              {user && !isAuthScreen && (
                <p className="text-xs text-gray-500">
                  Signed in as{" "}
                  <span className="font-medium text-agro-green">
                    {user.email}
                  </span>
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center justify-center rounded-full bg-agro-greenSoft text-agro-green w-9 h-9 text-xl">
            🌾
          </div>
        </header>

        {/* Body */}
        <main className="flex-1 px-4 pb-20 pt-2 overflow-y-auto">
          {children}
        </main>

        {/* Bottom Navigation */}
        {!isAuthScreen && (
          <nav className="fixed bottom-0 left-0 right-0 flex justify-center md:static">
            <div className="w-full max-w-md">
              <div className="mx-4 mb-3 rounded-full bg-white shadow-soft px-4 py-2 flex justify-between text-xs">
                {navItems.map((item) => {
                  const isActive = location.pathname === item.path;

                  return (
                    <button
                      key={item.path}
                      onClick={() => navigate(item.path)}
                      className={`flex flex-col items-center flex-1 py-1 rounded-full transition ${
                        isActive
                          ? "text-agro-green bg-agro-greenSoft"
                          : "text-gray-500"
                      }`}
                    >
                      <span className="text-lg leading-none">
                        {item.icon}
                      </span>
                      <span className="mt-0.5">{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </nav>
        )}
      </div>
    </div>
  );
};

export default MobileShell;
