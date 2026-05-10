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
      hasMasterCV: null, // null = unknown, true = has CV, false = no CV

      login: (userData, accessToken, refreshToken) => set({
        user: userData,
        token: accessToken,
        refreshToken,
        role: userData.role,
        isAuthenticated: true,
        hasMasterCV: null, // reset on login, will be resolved lazily
      }),

      setTokens: (accessToken, refreshToken) => set({
        token: accessToken,
        refreshToken,
      }),

      setHasMasterCV: (value) => set({ hasMasterCV: value }),

      logout: () => set({
        user: null,
        token: null,
        refreshToken: null,
        role: null,
        isAuthenticated: false,
        hasMasterCV: null,
      }),
    }),
    {
      name: 'auth_store',
    }
  )
);

export default useAuthStore;
