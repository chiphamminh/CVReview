import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      token: null,
      role: null, // 'HR' | 'CANDIDATE' | 'ADMIN'
      isAuthenticated: false,

      login: (userData, token) => set({
        user: userData,
        token: token,
        role: userData.role,
        isAuthenticated: true,
      }),

      logout: () => set({
        user: null,
        token: null,
        role: null,
        isAuthenticated: false,
      }),
    }),
    {
      name: 'auth_store', // Tên lưu trong localStorage
    }
  )
);

export default useAuthStore;
