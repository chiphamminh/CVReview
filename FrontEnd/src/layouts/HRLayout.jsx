import React, { useState } from 'react';
import { Layout, Menu, Button, theme, Avatar, Dropdown } from 'antd';
import {
  DashboardOutlined,
  UnorderedListOutlined,
  TeamOutlined,
  RobotOutlined,
  MenuUnfoldOutlined,
  MenuFoldOutlined,
  UserOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import useAuthStore from '@/store/authStore';
import { authApi } from '@/api/auth.api';

const { Header, Sider, Content } = Layout;

const HRLayout = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const { token: { borderRadiusLG } } = theme.useToken();
  const navigate = useNavigate();
  const location = useLocation();
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

  const menuItems = [
    {
      key: '/hr/dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/hr/positions',
      icon: <UnorderedListOutlined />,
      label: 'Positions',
    },
    {
      key: '/hr/candidates',
      icon: <TeamOutlined />,
      label: 'Candidates',
    },
    {
      key: '/hr/chatbot',
      icon: <RobotOutlined />,
      label: 'AI Chatbot',
    },
  ];

  const userMenu = {
    items: [
      {
        key: 'profile',
        icon: <UserOutlined />,
        label: 'Profile',
      },
      {
        key: 'logout',
        icon: <LogoutOutlined />,
        label: 'Logout',
        danger: true,
        onClick: handleLogout,
      },
    ],
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        theme="light"
        className="hr-sider"
        style={{
          background: 'linear-gradient(180deg, #0F172A 0%, #1E293B 100%)',
          boxShadow: '2px 0 12px rgba(0,0,0,0.18)',
        }}
      >
        <div style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 700,
          fontSize: collapsed ? 16 : 22,
          color: '#fff',
          letterSpacing: collapsed ? 0 : '0.03em',
          borderBottom: '1px solid rgba(148,163,184,0.12)',
          userSelect: 'none',
        }}>
          {collapsed ? 'CVR' : 'CV Review'}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          onClick={({ key }) => navigate(key)}
          items={menuItems}
          className="hr-sider-menu"
          style={{ borderRight: 0, fontSize: 15, background: 'transparent', marginTop: 8 }}
        />
      </Sider>
      <Layout>
        <Header style={{
          padding: '0 24px 0 0',
          background: '#fff',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid #E2E8F0',
          boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
          position: 'sticky',
          top: 0,
          zIndex: 100,
        }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />
          <div>
            <Dropdown menu={userMenu} placement="bottomRight" align={{ offset: [0, -8] }}>
              <div style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 8 }}>
                <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#4F46E5' }} />
                <span style={{ fontWeight: 500 }}> {user?.name || 'HR Admin'}</span>
              </div>
            </Dropdown>
          </div>
        </Header>
        <Content style={{
          margin: '16px 16px',
          padding: 5,
          minHeight: 280,
          background: '#F8FAFC',
          borderRadius: borderRadiusLG,
          overflow: 'auto',
        }}>
          <div key={location.pathname.split('/').slice(0, 3).join('/')} className="page-enter">
            {children}
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};

export default HRLayout;
