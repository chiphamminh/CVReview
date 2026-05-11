import axiosClient from './axiosClient';

export const analyticsApi = {
  getCvTraffic: (days = 30) =>
    axiosClient.get('/hr/analytics/cv-traffic', { params: { days } }),

  getOverview: (days = 30) =>
    axiosClient.get('/hr/analytics/overview', { params: { days } }),

  getScoreDistribution: () =>
    axiosClient.get('/hr/analytics/score-distribution'),
};
