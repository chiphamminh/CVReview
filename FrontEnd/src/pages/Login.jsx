import { useState, useCallback } from 'react';
import { Form, Input, Button, Tabs, Typography, Divider, App, Card } from 'antd';
import {
  PhoneOutlined,
  LockOutlined,
  UserOutlined,
  MailOutlined,
  RobotOutlined,
  GoogleOutlined,
  GithubOutlined,
  FacebookFilled,
} from '@ant-design/icons';
import { useNavigate, useSearchParams } from 'react-router-dom';
import useAuthStore from '@/store/authStore';
import { authApi } from '@/api/auth.api';

const { Title, Text } = Typography;

// ─── Constants ───────────────────────────────────────────────────────────────

const SOCIAL_PROVIDERS = [
  {
    key: 'google',
    label: 'Google',
    icon: <GoogleOutlined style={{ color: '#ea4335' }} />,
  },
  {
    key: 'github',
    label: 'GitHub',
    icon: <GithubOutlined style={{ color: '#24292e' }} />,
  },
  {
    key: 'facebook',
    label: 'Facebook',
    icon: <FacebookFilled style={{ color: '#1877f2' }} />,
  },
];

// ─── Social Login Section ─────────────────────────────────────────────────────

const SocialSection = ({ dividerText }) => {
  const { message } = App.useApp();

  const handleSocialLogin = useCallback(
    (providerKey) => {
      if (providerKey === 'google') {
        // TODO: redirect to OAuth2 Google endpoint
        message.info('Google login is coming soon.');
      } else {
        message.info(`${providerKey.charAt(0).toUpperCase() + providerKey.slice(1)} login is not available yet.`);
      }
    },
    [message]
  );

  return (
    <>
      <Divider style={{ color: '#8c8c8c', fontSize: 12, margin: '20px 0' }}>
        {dividerText}
      </Divider>
      <div style={{ display: 'flex', gap: 8 }}>
        {SOCIAL_PROVIDERS.map((p) => (
          <Button
            key={p.key}
            className="social-btn"
            icon={p.icon}
            onClick={() => handleSocialLogin(p.key)}
            style={{ flex: 1 }}
          >
            {p.label}
          </Button>
        ))}
      </div>
    </>
  );
};

// ─── Login Form ───────────────────────────────────────────────────────────────

const LoginForm = ({ onSuccess }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const { message } = App.useApp();
  const { login } = useAuthStore();

  const handleSubmit = useCallback(
    async (values) => {
      setLoading(true);
      try {
        const res = await authApi.login(values.phone, values.password);
        if (!res?.data) {
          message.error(res?.message || 'Invalid phone number or password.');
          return;
        }
        const { accessToken, refreshToken, account } = res.data;
        login(account, accessToken, refreshToken);
        message.success(`Welcome back, ${account.name}!`);
        onSuccess(account.role);
      } catch (err) {
        message.error(
          err.response?.data?.message || err.message || 'Login failed. Please try again.'
        );
      } finally {
        setLoading(false);
      }
    },
    [login, message, onSuccess]
  );

  return (
    <>
      <Form form={form} layout="vertical" onFinish={handleSubmit} size="large" style={{ marginTop: 4 }}>
        <Form.Item
          name="phone"
          label="Phone Number"
          rules={[
            { required: true, message: 'Please enter your phone number' },
            { pattern: /^[0-9]{9,11}$/, message: 'Invalid phone number format' },
          ]}
        >
          <Input
            prefix={<PhoneOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="0912 345 678"
          />
        </Form.Item>

        <Form.Item
          name="password"
          label="Password"
          rules={[{ required: true, message: 'Please enter your password' }]}
        >
          <Input.Password
            prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="••••••••"
          />
        </Form.Item>

        <Form.Item style={{ marginBottom: 0 }}>
          <Button
            type="primary"
            htmlType="submit"
            block
            loading={loading}
            style={{ height: 44, fontWeight: 500 }}
          >
            Sign In
          </Button>
        </Form.Item>
      </Form>

      <SocialSection dividerText="or sign in with" />
    </>
  );
};

// ─── Register Form ────────────────────────────────────────────────────────────

const RegisterForm = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const { message } = App.useApp();

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    // TODO: wire to POST /auth/register once BE is ready
    await new Promise((r) => setTimeout(r, 500));
    setLoading(false);
    message.info('Registration is coming soon. Please contact your administrator to get an account.');
    form.resetFields();
  }, [form, message]);

  return (
    <>
      <Form form={form} layout="vertical" onFinish={handleSubmit} size="large" style={{ marginTop: 4 }}>
        <Form.Item
          name="name"
          label="Full Name"
          rules={[{ required: true, message: 'Please enter your full name' }]}
        >
          <Input
            prefix={<UserOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="John Doe"
          />
        </Form.Item>

        <Form.Item
          name="phone"
          label="Phone Number"
          rules={[
            { required: true, message: 'Please enter your phone number' },
            { pattern: /^[0-9]{9,11}$/, message: 'Invalid phone number format' },
          ]}
        >
          <Input
            prefix={<PhoneOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="0912 345 678"
          />
        </Form.Item>

        <Form.Item
          name="email"
          label="Email"
          rules={[
            { required: true, message: 'Please enter your email' },
            { type: 'email', message: 'Invalid email format' },
          ]}
        >
          <Input
            prefix={<MailOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="you@example.com"
          />
        </Form.Item>

        <Form.Item
          name="password"
          label="Password"
          rules={[
            { required: true, message: 'Please enter a password' },
            { min: 8, message: 'Password must be at least 8 characters' },
          ]}
        >
          <Input.Password
            prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="••••••••"
          />
        </Form.Item>

        <Form.Item
          name="confirmPassword"
          label="Confirm Password"
          dependencies={['password']}
          rules={[
            { required: true, message: 'Please confirm your password' },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue('password') === value) {
                  return Promise.resolve();
                }
                return Promise.reject(new Error('Passwords do not match'));
              },
            }),
          ]}
        >
          <Input.Password
            prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="••••••••"
          />
        </Form.Item>

        <Form.Item style={{ marginBottom: 0 }}>
          <Button
            type="primary"
            htmlType="submit"
            block
            loading={loading}
            style={{ height: 44, fontWeight: 500 }}
          >
            Create Account
          </Button>
        </Form.Item>
      </Form>

      <SocialSection dividerText="or sign up with" />
    </>
  );
};

// ─── Main Page ────────────────────────────────────────────────────────────────

const Login = () => {
  const [searchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'login');
  const navigate = useNavigate();

  const handleLoginSuccess = useCallback(
    (role) => {
      if (role === 'HR' || role === 'ADMIN') navigate('/hr/positions');
      else navigate('/candidate/cv');
    },
    [navigate]
  );

  const tabItems = [
    {
      key: 'login',
      label: 'Sign In',
      children: <LoginForm onSuccess={handleLoginSuccess} />,
    },
    {
      key: 'register',
      label: 'Sign Up',
      children: <RegisterForm />,
    },
  ];

  return (
    <div className="auth-bg">
      <div className="auth-card">
        <Card
          variant="borderless"
          style={{
            borderRadius: 16,
            boxShadow: '0 24px 64px rgba(0, 0, 0, 0.10)',
            padding: '8px 4px',
          }}
        >
          {/* Brand header */}
          <div style={{ textAlign: 'center', marginBottom: 8, paddingTop: 4 }}>
            <div
              style={{
                width: 52,
                height: 52,
                borderRadius: 14,
                background: 'linear-gradient(135deg, #1677ff, #0958d9)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontSize: 24,
                marginBottom: 12,
                boxShadow: '0 8px 20px rgba(22, 119, 255, 0.3)',
              }}
            >
              <RobotOutlined />
            </div>
            <Title level={4} style={{ margin: '0 0 2px', color: '#141414' }}>
              CV Review
            </Title>
            <Text type="secondary" style={{ fontSize: 13 }}>
              AI-powered recruitment platform
            </Text>
          </div>

          <Tabs
            centered
            activeKey={activeTab}
            onChange={setActiveTab}
            items={tabItems}
            style={{ marginTop: 4 }}
          />
        </Card>
      </div>
    </div>
  );
};

export default Login;
