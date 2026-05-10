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

const { Header, Sider, Content } = Layout;

const HRLayout = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const { token: { colorBgContainer, borderRadiusLG } } = theme.useToken();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
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
      <Sider trigger={null} collapsible collapsed={collapsed} theme="light" style={{ borderRight: '1px solid #f0f0f0' }}>
        <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: collapsed ? 16 : 20, color: '#1677ff', borderBottom: '1px solid #f0f0f0' }}>
          {collapsed ? 'CVR' : 'CV Review'}
        </div>
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[location.pathname]}
          onClick={({ key }) => navigate(key)}
          items={menuItems}
          style={{ borderRight: 0, fontSize: 16 }}
        />
      </Sider>
      <Layout>
        <Header style={{ padding: '0 24px 0 0', background: colorBgContainer, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #f0f0f0' }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />
          <div>
            <Dropdown menu={userMenu} placement="bottomRight" align={{ offset: [0, -8] }}>
              <div style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', gap: 8 }}>
                <span style={{ fontWeight: 500 }}>{user?.name || 'HR Admin'}</span>
                <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#1677ff' }} />
              </div>
            </Dropdown>
          </div>
        </Header>
        <Content style={{ margin: '16px 16px', padding: 24, minHeight: 280, background: colorBgContainer, borderRadius: borderRadiusLG, overflow: 'auto' }}>
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default HRLayout;
