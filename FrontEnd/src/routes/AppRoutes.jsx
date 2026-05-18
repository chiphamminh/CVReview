import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { Spin } from 'antd';
import useAuthStore from '@/store/authStore';

// Layouts & Protected Routes
import HRLayout from '@/layouts/HRLayout';
import ProtectedRoute from '@/components/common/ProtectedRoute';

// Pages
import Login from '@/pages/Login';
import ForgotPasswordPage from '@/pages/ForgotPasswordPage';
import PositionsPage from '@/pages/hr/PositionsPage';
import CandidatesPage from '@/pages/hr/CandidatesPage';
import HRChatbotPage from '@/pages/hr/HRChatbotPage';

// Lazy-load dashboard to code-split the heavy @ant-design/plots bundle
const HRDashboardPage = lazy(() => import('@/pages/hr/HRDashboardPage'));

import CandidateLayout from '@/layouts/CandidateLayout';
import CareerPage from '@/pages/candidate/CareerPage';
import CVPage from '@/pages/candidate/CVPage';

const PageLoader = () => (
  <div style={{ height: '60vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Spin size="large" />
  </div>
);

const NotFound = () => <div>404 - Not Found</div>;

const AppRoutes = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        
        {/* Candidate Layout (includes Public Careers and Protected Routes) */}
        <Route path="/" element={<CandidateLayout><Outlet /></CandidateLayout>}>
          <Route index element={<Navigate to="/careers" replace />} />
          <Route path="careers" element={<CareerPage />} />
          
          <Route path="candidate" element={<ProtectedRoute allowedRoles={['CANDIDATE']}><Outlet /></ProtectedRoute>}>
            <Route path="cv" element={<CVPage />} />
            <Route path="*" element={<Navigate to="cv" replace />} />
          </Route>
        </Route>

        {/* HR Routes */}
        <Route
          path="/hr/*"
          element={
            <ProtectedRoute allowedRoles={['HR', 'ADMIN']}>
              <HRLayout>
                <Routes>
                  <Route path="dashboard" element={<Suspense fallback={<PageLoader />}><HRDashboardPage /></Suspense>} />
                  <Route path="positions" element={<PositionsPage />} />
                  <Route path="candidates" element={<CandidatesPage />} />
                  <Route path="chatbot/:positionId?" element={<HRChatbotPage />} />
                  <Route path="*" element={<Navigate to="positions" replace />} />
                </Routes>
              </HRLayout>
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
