import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      token: null,
      refreshToken: null,
      role: null,
      isAuthenticated: false,

      login: (userData, accessToken, refreshToken) => set({
        user: userData,
        token: accessToken,
        refreshToken,
        role: userData.role,
        isAuthenticated: true,
      }),

      setTokens: (accessToken, refreshToken) => set({
        token: accessToken,
        refreshToken,
      }),

      logout: () => set({
        user: null,
        token: null,
        refreshToken: null,
        role: null,
        isAuthenticated: false,
      }),
    }),
    {
      name: 'auth_store',
    }
  )
);

export default useAuthStore;
