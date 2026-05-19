import { create } from 'zustand';

/**
 * In-memory store for HR Dashboard filter state.
 * Survives SPA navigation; clears on page refresh / logout.
 */
const useDashboardStore = create((set) => ({
  days: 30,
  selectedPositionId: null,

  setDays: (days) => set({ days }),
  setSelectedPositionId: (id) => set({ selectedPositionId: id }),
}));

export default useDashboardStore;
