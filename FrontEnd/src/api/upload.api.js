import axiosClient from './axiosClient';

export const uploadApi = {
  hrUploadCVs: (positionId, files) => {
    const fd = new FormData();
    fd.append('positionId', positionId);
    files.forEach((f) => fd.append('files', f));
    return axiosClient.post('/upload/hr/cv', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  candidateUploadCV: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return axiosClient.post('/upload/candidate/cv', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  getBatchStatus: (batchId) =>
    axiosClient.get(`/tracking/${batchId}/status`),
};
