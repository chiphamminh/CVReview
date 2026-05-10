import { create } from 'zustand';

const useUiStore = create((set) => ({
  chatbotOpen: false,

  openChatbot: () => set({ chatbotOpen: true }),
  closeChatbot: () => set({ chatbotOpen: false }),
}));

export default useUiStore;
