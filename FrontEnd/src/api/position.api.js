import axiosClient from './axiosClient';

export const positionApi = {
  filter: (params) =>
    axiosClient.get('/positions', { params }),

  create: (formData) =>
    axiosClient.post('/positions', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  update: (id, formData) =>
    axiosClient.put(`/positions/${id}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  // score is @RequestParam on the BE, not request body
  updateMinScore: (id, score) =>
    axiosClient.patch(`/positions/${id}/minimum-fit-score`, null, { params: { score } }),

  toggleActive: (id) =>
    axiosClient.patch(`/positions/${id}/toggle-active`),

  // BE accepts List<Integer> directly as body
  deleteMany: (ids) =>
    axiosClient.delete('/positions', { data: ids }),

  getJDText: (id) =>
    axiosClient.get(`/positions/jd/${id}/text`),
};
