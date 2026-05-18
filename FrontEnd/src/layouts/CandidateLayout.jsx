import React from 'react';
import { Layout, Menu, Button, Dropdown, Space, Avatar, Typography } from 'antd';
import { UserOutlined, RobotOutlined, FileTextOutlined, LogoutOutlined } from '@ant-design/icons';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import useAuthStore from '@/store/authStore';
import useUiStore from '@/store/uiStore';
import { authApi } from '@/api/auth.api';
import CandidateChatbotDrawer from '@/components/chatbot/CandidateChatbotDrawer';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

const CandidateLayout = ({ children }) => {
  const { chatbotOpen, openChatbot, closeChatbot } = useUiStore();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { user, logout, refreshToken } = useAuthStore();

  const handleLogout = async () => {
    try {
      await authApi.logout(refreshToken);
    } catch {
      // best-effort: always clear local state even if API fails
    } finally {
      logout();
      navigate('/login');
    }
  };

  const userMenu = {
    items: [
      {
        key: 'cv',
        icon: <FileTextOutlined />,
        label: <Link to="/candidate/cv">My CV</Link>,
      },
      {
        type: 'divider',
      },
      {
        key: 'logout',
        icon: <LogoutOutlined />,
        label: 'Logout',
        onClick: handleLogout,
      },
    ],
  };

  return (
    <Layout style={{ minHeight: '100vh', backgroundColor: '#f0f2f5' }}>
      <Header style={{
        position: 'sticky',
        top: 0,
        zIndex: 1000,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#fff',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <div className="logo" style={{ cursor: 'pointer' }} onClick={() => navigate('/careers')}>
            <Title level={3} style={{ margin: 0, color: '#1677ff' }}>TechCorp</Title>
          </div>
          <Menu
            mode="horizontal"
            selectedKeys={[pathname]}
            style={{ borderBottom: 'none', minWidth: '400px' }}
            items={[
              { key: '/careers', label: <Link to="/careers">Careers</Link> },
              { key: '/candidate/cv', label: <Link to="/candidate/cv">My CV</Link> },
            ]}
          />
        </div>

        <div>
          {user ? (
            <Dropdown menu={userMenu} placement="bottomRight" arrow align={{ offset: [0, -8] }} >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#1677ff' }} />
                <span style={{ fontWeight: 500 }}>{user.name || 'Candidate'}</span>
              </Space>
            </Dropdown>
          ) : (
            <Space>
              <Button type="text" onClick={() => navigate('/login')}>Login</Button>
              <Button type="primary" onClick={() => navigate('/login')}>Sign Up</Button>
            </Space>
          )}
        </div>
      </Header>

      <Content style={{ padding: '0 50px', marginTop: 24 }}>
        <div style={{ minHeight: 380, padding: 24, background: '#fff', borderRadius: 8 }}>
          {children}
        </div>
      </Content>

      <Footer style={{ textAlign: 'center' }}>
        TechCorp Recruitment ©{new Date().getFullYear()} - Connecting Talent with Opportunities
      </Footer>

      {/* Floating Action Button for Chatbot */}
      <div
        style={{
          position: 'fixed',
          bottom: 40,
          right: 40,
          zIndex: 999
        }}
      >
        <Button
          type="primary"
          shape="circle"
          icon={<RobotOutlined style={{ fontSize: 24 }} />}
          size="large"
          style={{ width: 60, height: 60, boxShadow: '0 4px 12px rgba(22,119,255,0.4)' }}
          onClick={openChatbot}
        />
      </div>

      <CandidateChatbotDrawer
        open={chatbotOpen}
        onClose={closeChatbot}
      />
    </Layout>
  );
};

export default CandidateLayout;
