import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import useAuthStore from '@/store/authStore';

// Layouts & Protected Routes
import HRLayout from '@/layouts/HRLayout';
import ProtectedRoute from '@/components/common/ProtectedRoute';

// Pages
import Login from '@/pages/Login';
import PositionsPage from '@/pages/hr/PositionsPage';
import CandidatesPage from '@/pages/hr/CandidatesPage';
import HRChatbotPage from '@/pages/hr/HRChatbotPage';

// Placeholders
const CandidateLayout = ({ children }) => <div><h2>Candidate Layout</h2>{children}</div>;
const HRDashboard = () => <div>HR Dashboard</div>;
const Careers = () => <div>Careers Page</div>;
const CandidateCV = () => <div>Candidate CV Page</div>;
const NotFound = () => <div>404 - Not Found</div>;

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
                  <Route path="positions" element={<PositionsPage />} />
                  <Route path="candidates" element={<CandidatesPage />} />
                  <Route path="chatbot/:positionId?" element={<HRChatbotPage />} />
                  <Route path="*" element={<Navigate to="positions" replace />} />
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
