import axiosClient from './axiosClient';

export const candidateApi = {
  filter: (params) => axiosClient.get('/cv/candidates', { params }),
  getById: (cvId) => axiosClient.get(`/cv/${cvId}`),
  getMyCV: () => axiosClient.get('/cv/me'),
  getMyApplications: () => axiosClient.get('/cv/my-applications'),
  deleteMany: (ids) => axiosClient.delete('/cv', { data: ids }),
  updateInfo: (cvId, params) => axiosClient.put(`/cv/${cvId}`, null, { params }),
  scheduleInterview: (cvId, data) => axiosClient.post(`/cv/${cvId}/schedule-interview`, data),
  rescheduleInterview: (cvId, data) => axiosClient.post(`/cv/${cvId}/reschedule-interview`, data),
  sendOffer: (cvId, data) => {
    const fd = new FormData();
    fd.append('startDate', data.startDate);
    fd.append('offerExpirationDate', data.offerExpirationDate);
    (data.files || []).forEach(f => {
      if (f.originFileObj) fd.append('attachments', f.originFileObj);
    });
    return axiosClient.post(`/cv/${cvId}/send-offer`, fd);
  },
  updateStage: (cvId, stage) => axiosClient.patch(`/cv/${cvId}/stage`, { recruitmentStage: stage }),
  getFailedBatches: () => axiosClient.get('/cv/failed-batches'),
  deleteFailedBatches: (batchIds) => axiosClient.delete('/cv/failed-batches', { data: batchIds }),
};
