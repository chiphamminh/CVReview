import axiosClient from './axiosClient';

export const authApi = {
  login: (phone, password) =>
    axiosClient.post('/auth/login', { phone, password }),

  logout: () =>
    axiosClient.post('/auth/logout'),

  refreshToken: (refreshToken) =>
    axiosClient.post('/auth/refresh-token', { refreshToken }),
};
