import { useState, useCallback, useRef } from 'react';
import { Form, Input, Button, Typography, Card, App, Steps } from 'antd';
import {
  MailOutlined,
  LockOutlined,
  RobotOutlined,
  ArrowLeftOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { authApi } from '@/api/auth.api';

const { Title, Text } = Typography;

const RESEND_COOLDOWN = 60; // seconds

const STEPS = ['Email', 'OTP', 'New Password'];

const ForgotPasswordPage = () => {
  const [currentStep, setCurrentStep] = useState(0); // 0=EMAIL, 1=OTP, 2=RESET
  const [emailForm] = Form.useForm();
  const [otpForm] = Form.useForm();
  const [resetForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [email, setEmail] = useState('');
  const [resetToken, setResetToken] = useState('');
  const { message } = App.useApp();
  const navigate = useNavigate();
  const cooldownRef = useRef(null);

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

  // Step 0 — Submit email
  const handleEmailSubmit = useCallback(
    async (values) => {
      setLoading(true);
      try {
        const normalizedEmail = values.email.trim().toLowerCase();
        await authApi.forgotPassword(normalizedEmail);
        setEmail(normalizedEmail);
        setCurrentStep(1);
        startCooldown();
        // Always show same message (anti-enumeration — mirrors BE behavior)
        message.success('If this email is registered, an OTP has been sent.');
      } catch (err) {
        // BE always returns 200, so this only fires on network errors
        message.error(err.response?.data?.message || err.message || 'Something went wrong.');
      } finally {
        setLoading(false);
      }
    },
    [startCooldown, message]
  );

  // Step 1 — Verify OTP
  const handleOtpSubmit = useCallback(
    async (values) => {
      setLoading(true);
      try {
        const res = await authApi.verifyResetOtp(email, values.otp);
        if (!res?.data?.resetToken) {
          throw new Error(res?.message || 'OTP verification failed');
        }
        setResetToken(res.data.resetToken);
        setCurrentStep(2);
      } catch (err) {
        message.error(
          err.response?.data?.message || err.message || 'Invalid or expired OTP.'
        );
      } finally {
        setLoading(false);
      }
    },
    [email, message]
  );

  // Step 1 — Resend OTP
  const handleResend = useCallback(async () => {
    if (cooldown > 0) return;
    setResendLoading(true);
    try {
      await authApi.forgotPassword(email);
      startCooldown();
      message.success('OTP resent! Please check your email.');
      otpForm.resetFields();
    } catch (err) {
      message.error(err.response?.data?.message || err.message || 'Failed to resend OTP.');
    } finally {
      setResendLoading(false);
    }
  }, [email, cooldown, startCooldown, message, otpForm]);

  // Step 2 — Reset password
  const handleResetSubmit = useCallback(
    async (values) => {
      setLoading(true);
      try {
        const res = await authApi.resetPassword(resetToken, values.newPassword);
        if (res?.statusCode !== undefined && res.statusCode !== 200 && res.statusCode !== 0) {
          throw new Error(res?.message || 'Password reset failed');
        }
        message.success('Password reset successful! Please sign in with your new password.');
        navigate('/login');
      } catch (err) {
        message.error(
          err.response?.data?.message || err.message || 'Password reset failed. Please try again.'
        );
      } finally {
        setLoading(false);
      }
    },
    [resetToken, message, navigate]
  );

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
          <div style={{ textAlign: 'center', marginBottom: 24, paddingTop: 4 }}>
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
              Forgot Password
            </Title>
            <Text type="secondary" style={{ fontSize: 13 }}>
              Reset your CV Review account password
            </Text>
          </div>

          <Steps
            current={currentStep}
            items={STEPS.map((s) => ({ title: s }))}
            size="small"
            style={{ marginBottom: 28 }}
          />

          {/* ── Step 0: Email ─────────────────────────────────────────────── */}
          {currentStep === 0 && (
            <Form form={emailForm} layout="vertical" onFinish={handleEmailSubmit} size="large">
              <Text type="secondary" style={{ display: 'block', marginBottom: 16, fontSize: 13 }}>
                Enter your registered email address and we'll send you an OTP.
              </Text>

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

              <Form.Item style={{ marginBottom: 12 }}>
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

              <div style={{ textAlign: 'center' }}>
                <Button
                  type="text"
                  icon={<ArrowLeftOutlined />}
                  onClick={() => navigate('/login')}
                  style={{ color: '#595959', fontSize: 13 }}
                >
                  Back to Sign In
                </Button>
              </div>
            </Form>
          )}

          {/* ── Step 1: OTP ───────────────────────────────────────────────── */}
          {currentStep === 1 && (
            <Form form={otpForm} layout="vertical" onFinish={handleOtpSubmit} size="large">
              <Button
                type="text"
                icon={<ArrowLeftOutlined />}
                onClick={() => { otpForm.resetFields(); setCurrentStep(0); clearInterval(cooldownRef.current); setCooldown(0); }}
                style={{ padding: 0, marginBottom: 16, color: '#595959' }}
              >
                Back
              </Button>
              <Text type="secondary" style={{ display: 'block', marginBottom: 16, fontSize: 13 }}>
                We sent a 6-digit OTP to <strong>{email}</strong>. Enter it below.
              </Text>

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
                  Verify OTP
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
          )}

          {/* ── Step 2: New Password ──────────────────────────────────────── */}
          {currentStep === 2 && (
            <Form form={resetForm} layout="vertical" onFinish={handleResetSubmit} size="large">
              <Text type="secondary" style={{ display: 'block', marginBottom: 16, fontSize: 13 }}>
                Choose a strong new password for your account.
              </Text>

              <Form.Item
                name="newPassword"
                label="New Password"
                rules={[
                  { required: true, message: 'Please enter a new password' },
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
                dependencies={['newPassword']}
                rules={[
                  { required: true, message: 'Please confirm your password' },
                  ({ getFieldValue }) => ({
                    validator(_, value) {
                      if (!value || getFieldValue('newPassword') === value) return Promise.resolve();
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
                  Reset Password
                </Button>
              </Form.Item>
            </Form>
          )}
        </Card>
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
