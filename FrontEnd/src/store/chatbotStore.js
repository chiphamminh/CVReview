import { create } from 'zustand';

/**
 * In-memory store for HR Chatbot state.
 * Not persisted to localStorage — clears on page refresh / logout redirect.
 * Only survives SPA navigation (user goes to another page and comes back).
 */
const useChatbotStore = create((set) => ({
  selectedPositionId: null,
  currentSessionId: null,
  mode: 'Internal',
  sidebarCollapsed: false,

  setSelectedPositionId: (id) => set({ selectedPositionId: id }),
  setCurrentSessionId: (id) => set({ currentSessionId: id }),
  setMode: (m) => set({ mode: m }),
  setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
}));

export default useChatbotStore;
