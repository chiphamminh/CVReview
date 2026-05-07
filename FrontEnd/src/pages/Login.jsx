import React from 'react';
import { Button, Card, Typography, Space } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import useAuthStore from '@/store/authStore';
import { useNavigate } from 'react-router-dom';

const { Title, Text } = Typography;

const Login = () => {
  const { login } = useAuthStore();
  const navigate = useNavigate();

  const handleFakeLoginHR = () => {
    login({ name: 'HR Admin', role: 'HR' }, 'fake-jwt-token-hr');
    navigate('/hr/dashboard');
  };

  const handleFakeLoginCandidate = () => {
    login({ name: 'Ứng viên Demo', role: 'CANDIDATE' }, 'fake-jwt-token-candidate');
    navigate('/candidate/cv');
  };

  return (
    <div style={{ height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#f5f5f5' }}>
      <Card style={{ width: 400, textAlign: 'center', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', borderRadius: 12 }}>
        <Title level={3} style={{ marginTop: 0 }}>Hệ thống Tuyển dụng AI</Title>
        <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
          Môi trường phát triển (Dev) - Vui lòng chọn quyền đăng nhập giả lập.
        </Text>

        <Space direction="vertical" size="middle" style={{ display: 'flex' }}>
          <Button 
            type="primary" 
            size="large" 
            icon={<UserOutlined />} 
            block 
            onClick={handleFakeLoginHR}
          >
            Đăng nhập quyền HR (Quản trị)
          </Button>
          
          <Button 
            size="large" 
            icon={<RobotOutlined />} 
            block 
            onClick={handleFakeLoginCandidate}
          >
            Đăng nhập quyền Ứng viên
          </Button>
        </Space>
      </Card>
    </div>
  );
};

export default Login;
