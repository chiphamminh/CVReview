import { useState, useCallback, useRef, useEffect } from 'react';
import { Form, Input, Button, Tabs, Typography, Divider, App, Card } from 'antd';
import {
  LockOutlined,
  UserOutlined,
  MailOutlined,
  RobotOutlined,
  GoogleOutlined,
  GithubOutlined,
  FacebookFilled,
  ArrowLeftOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import useAuthStore from '@/store/authStore';
import { authApi } from '@/api/auth.api';

const { Title, Text } = Typography;

// ─── Constants ───────────────────────────────────────────────────────────────

const SOCIAL_PROVIDERS = [
  { key: 'google', label: 'Google', icon: <GoogleOutlined style={{ color: '#ea4335' }} /> },
  { key: 'github', label: 'GitHub', icon: <GithubOutlined style={{ color: '#24292e' }} /> },
  { key: 'facebook', label: 'Facebook', icon: <FacebookFilled style={{ color: '#1877f2' }} /> },
];

// ─── Social Login Section ─────────────────────────────────────────────────────

const SocialSection = ({ dividerText }) => {
  const { message } = App.useApp();

  const handleSocialLogin = useCallback(
    (providerKey) => {
      if (providerKey === 'google') {
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
        const identifier = values.identifier?.trim();
        const isEmail = identifier.includes('@');

        let res;
        if (isEmail) {
          res = await authApi.candidateLogin(identifier, values.password);
        } else {
          res = await authApi.login(identifier, values.password);
        }

        if (!res?.data) {
          message.error(res?.message || 'Invalid credentials.');
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
          name="identifier"
          label="Phone or Email"
          rules={[{ required: true, message: 'Please enter your phone number or email' }]}
        >
          <Input
            prefix={<MailOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="0912345678 or you@example.com"
          />
        </Form.Item>

        <Form.Item
          name="password"
          label="Password"
          rules={[{ required: true, message: 'Please enter your password' }]}
          style={{ marginBottom: 4 }}
        >
          <Input.Password
            prefix={<LockOutlined style={{ color: '#bfbfbf' }} />}
            placeholder="••••••••"
          />
        </Form.Item>

        <div style={{ textAlign: 'right', marginBottom: 16 }}>
          <Link to="/forgot-password" style={{ fontSize: 13, color: '#4F46E5' }}>
            Forgot password?
          </Link>
        </div>

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

const RESEND_COOLDOWN = 60; // seconds

const RegisterForm = ({ onRegistered }) => {
  const [form] = Form.useForm();
  const [otpForm] = Form.useForm();
  const [step, setStep] = useState('form'); // 'form' | 'otp'
  const [loading, setLoading] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [pendingData, setPendingData] = useState(null); // { email, name, password }
  const { message } = App.useApp();
  const cooldownRef = useRef(null);

  // Clear interval on unmount to prevent memory leaks
  useEffect(() => () => clearInterval(cooldownRef.current), []);

  const startCooldown = useCallback(() => {
    setCooldown(RESEND_COOLDOWN);
    cooldownRef.current = setInterval(() => {
      setCooldown((c) => {
        if (c <= 1) {
          clearInterval(cooldownRef.current);
          return 0;
        }
        return c - 1;
      });
    }, 1000);
  }, []);

  const sendOtp = useCallback(
    async (email, name, password) => {
      const res = await authApi.candidateRegister(email, name, password);
      if (res?.statusCode !== undefined && res.statusCode !== 200 && res.statusCode !== 0) {
        throw new Error(res?.message || 'Failed to send OTP');
      }
      return res;
    },
    []
  );

  const handleFormSubmit = useCallback(
    async (values) => {
      setLoading(true);
      try {
        await sendOtp(values.email.trim().toLowerCase(), values.name.trim(), values.password);
        setPendingData({
          email: values.email.trim().toLowerCase(),
          name: values.name.trim(),
          password: values.password,
        });
        setStep('otp');
        startCooldown();
        message.success('OTP sent! Please check your email.');
      } catch (err) {
        message.error(
          err.response?.data?.message || err.message || 'Failed to send OTP. Please try again.'
        );
      } finally {
        setLoading(false);
      }
    },
    [sendOtp, startCooldown, message]
  );

  const handleOtpSubmit = useCallback(
    async (values) => {
      setLoading(true);
      try {
        const res = await authApi.verifyRegister(
          pendingData.email,
          values.otp,
          pendingData.name,
          pendingData.password
        );
        if (res?.statusCode !== undefined && res.statusCode !== 200 && res.statusCode !== 0) {
          throw new Error(res?.message || 'OTP verification failed');
        }
        message.success('Registration successful! Please sign in.');
        onRegistered();
      } catch (err) {
        message.error(
          err.response?.data?.message || err.message || 'Invalid or expired OTP.'
        );
      } finally {
        setLoading(false);
      }
    },
    [pendingData, message, onRegistered]
  );

  const handleResend = useCallback(async () => {
    if (!pendingData || cooldown > 0) return;
    setResendLoading(true);
    try {
      await sendOtp(pendingData.email, pendingData.name, pendingData.password);
      startCooldown();
      message.success('OTP resent! Please check your email.');
      otpForm.resetFields();
    } catch (err) {
      message.error(
        err.response?.data?.message || err.message || 'Failed to resend OTP.'
      );
    } finally {
      setResendLoading(false);
    }
  }, [pendingData, cooldown, sendOtp, startCooldown, message, otpForm]);

  const handleBack = useCallback(() => {
    clearInterval(cooldownRef.current);
    setCooldown(0);
    setStep('form');
    otpForm.resetFields();
  }, [otpForm]);

  if (step === 'otp') {
    return (
      <div style={{ marginTop: 4 }}>
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          onClick={handleBack}
          style={{ padding: 0, marginBottom: 16, color: '#595959' }}
        >
          Back
        </Button>

        <Text type="secondary" style={{ display: 'block', marginBottom: 20, fontSize: 13 }}>
          We sent a 6-digit OTP to <strong>{pendingData?.email}</strong>. Enter it below to complete registration.
        </Text>

        <Form form={otpForm} layout="vertical" onFinish={handleOtpSubmit} size="large">
          <Form.Item
            name="otp"
            label="OTP Code"
            rules={[
              { required: true, message: 'Please enter the OTP' },
              { pattern: /^\d{6}$/, message: 'OTP must be 6 digits' },
            ]}
          >
            <Input
              placeholder="123456"
              maxLength={6}
              style={{ letterSpacing: 6, fontWeight: 600, fontSize: 18 }}
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 8 }}>
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
              style={{ height: 44, fontWeight: 500 }}
            >
              Verify & Create Account
            </Button>
          </Form.Item>

          <div style={{ textAlign: 'center' }}>
            <Button
              type="text"
              icon={<ReloadOutlined />}
              onClick={handleResend}
              loading={resendLoading}
              disabled={cooldown > 0}
              style={{ color: cooldown > 0 ? '#bfbfbf' : '#4F46E5', fontSize: 13 }}
            >
              {cooldown > 0 ? `Resend OTP in ${cooldown}s` : 'Resend OTP'}
            </Button>
          </div>
        </Form>
      </div>
    );
  }

  return (
    <>
      <Form form={form} layout="vertical" onFinish={handleFormSubmit} size="large" style={{ marginTop: 4 }}>
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
                if (!value || getFieldValue('password') === value) return Promise.resolve();
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
            Send OTP
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
  const [registerKey, setRegisterKey] = useState(0);
  const navigate = useNavigate();

  const handleLoginSuccess = useCallback(
    (role) => {
      if (role === 'HR' || role === 'ADMIN') navigate('/hr/positions');
      else navigate('/candidate/cv');
    },
    [navigate]
  );

  const handleRegistered = useCallback(() => {
    setActiveTab('login');
    setRegisterKey((k) => k + 1); // force remount → reset form state + clear interval
  }, []);

  const tabItems = [
    {
      key: 'login',
      label: 'Sign In',
      children: <LoginForm onSuccess={handleLoginSuccess} />,
    },
    {
      key: 'register',
      label: 'Sign Up',
      children: <RegisterForm key={registerKey} onRegistered={handleRegistered} />,
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
