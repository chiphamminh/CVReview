import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import useAuthStore from '@/store/authStore';

// Layouts (Placeholders)
const HRLayout = ({ children }) => <div><h2>HR Layout</h2>{children}</div>;
const CandidateLayout = ({ children }) => <div><h2>Candidate Layout</h2>{children}</div>;

// Pages (Placeholders)
const Login = () => <div>Login Page</div>;
const HRDashboard = () => <div>HR Dashboard</div>;
const Careers = () => <div>Careers Page</div>;
const CandidateCV = () => <div>Candidate CV Page</div>;
const NotFound = () => <div>404 - Not Found</div>;

import ProtectedRoute from '@/components/common/ProtectedRoute';

const AppRoutes = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/careers" element={<Careers />} />
        <Route path="/" element={<Navigate to="/careers" replace />} />

        {/* HR Routes */}
        <Route
          path="/hr/*"
          element={
            <ProtectedRoute allowedRoles={['HR', 'ADMIN']}>
              <HRLayout>
                <Routes>
                  <Route path="dashboard" element={<HRDashboard />} />
                  <Route path="*" element={<Navigate to="dashboard" replace />} />
                </Routes>
              </HRLayout>
            </ProtectedRoute>
          }
        />

        {/* Candidate Routes */}
        <Route
          path="/candidate/*"
          element={
            <ProtectedRoute allowedRoles={['CANDIDATE']}>
              <CandidateLayout>
                <Routes>
                  <Route path="cv" element={<CandidateCV />} />
                  <Route path="*" element={<Navigate to="cv" replace />} />
                </Routes>
              </CandidateLayout>
            </ProtectedRoute>
          }
        />

        {/* Catch All */}
        <Route path="/unauthorized" element={<div>403 - Unauthorized</div>} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
};

export default AppRoutes;
