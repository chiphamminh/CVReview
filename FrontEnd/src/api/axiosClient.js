import axios from 'axios';

const axiosClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8080/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor
axiosClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token'); // Hoặc lấy từ Zustand store nếu được serialize
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response Interceptor
axiosClient.interceptors.response.use(
  (response) => {
    // Trả về trực tiếp data
    return response.data;
  },
  (error) => {
    // Xử lý lỗi toàn cục
    if (error.response?.status === 401) {
      // Token hết hạn hoặc không hợp lệ -> Logout
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_store'); // Zustand persist key
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default axiosClient;
