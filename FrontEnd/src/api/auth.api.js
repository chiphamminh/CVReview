import axiosClient from './axiosClient';

export const authApi = {
  login: (phone, password) =>
    axiosClient.post('/auth/login', { phone, password }),

  logout: (refreshToken) =>
    axiosClient.post('/auth/logout', { refreshToken }),

  refreshToken: (refreshToken) =>
    axiosClient.post('/auth/refresh-token', { refreshToken }),

  // ─── Candidate ───────────────────────────────────────────────────────────────

  candidateRegister: (email, name, password) =>
    axiosClient.post('/auth/candidate/register', { email, name, password }),

  verifyRegister: (email, otp, name, password) =>
    axiosClient.post('/auth/candidate/verify-register', { email, otp, name, password }),

  candidateLogin: (email, password) =>
    axiosClient.post('/auth/candidate/login', { email, password }),

  forgotPassword: (email) =>
    axiosClient.post('/auth/candidate/forgot-password', { email }),

  verifyResetOtp: (email, otp) =>
    axiosClient.post('/auth/candidate/verify-reset-otp', { email, otp }),

  resetPassword: (resetToken, newPassword) =>
    axiosClient.post('/auth/candidate/reset-password', { resetToken, newPassword }),
};
