import { create } from 'zustand';

const useCandidateStore = create((set) => ({
  searchInput: '',
  keyword: '',
  stageFilter: null,
  positionFilter: null,
  typeFilter: null,
  isScoredFilter: null,
  scoreSort: null,
  page: 0,
  pageSize: 10,

  setSearchInput: (searchInput) => set({ searchInput }),
  setKeyword: (keyword) => set({ keyword, page: 0 }),
  setStageFilter: (stageFilter) => set({ stageFilter, page: 0 }),
  setPositionFilter: (positionFilter) => set({ positionFilter, page: 0 }),
  setTypeFilter: (typeFilter) => set({ typeFilter, page: 0 }),
  setIsScoredFilter: (isScoredFilter) => set({ isScoredFilter, page: 0 }),
  setScoreSort: (scoreSort) => set({ scoreSort, page: 0 }),
  setPagination: (page, pageSize) => set({ page, pageSize }),

  clearAllFilters: () =>
    set({
      searchInput: '',
      keyword: '',
      stageFilter: null,
      positionFilter: null,
      typeFilter: null,
      isScoredFilter: null,
      scoreSort: null,
      page: 0,
    }),
}));

export default useCandidateStore;
