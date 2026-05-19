import axiosClient from './axiosClient';

const buildParams = (base, positionId) =>
  positionId != null ? { ...base, positionId } : base;

export const analyticsApi = {
  getActivePositions: () =>
    axiosClient.get('/hr/analytics/active-positions'),

  getCvTraffic: (days = 30, positionId = null) =>
    axiosClient.get('/hr/analytics/cv-traffic', { params: buildParams({ days }, positionId) }),

  getOverview: (days = 30, positionId = null) =>
    axiosClient.get('/hr/analytics/overview', { params: buildParams({ days }, positionId) }),

  getScoreDistribution: (positionId = null) =>
    axiosClient.get('/hr/analytics/score-distribution', { params: buildParams({}, positionId) }),

  getStagePipeline: (positionId = null) =>
    axiosClient.get('/hr/analytics/stage-pipeline', { params: buildParams({}, positionId) }),

  getSourceBreakdown: (days = 30, positionId = null) =>
    axiosClient.get('/hr/analytics/source-breakdown', { params: buildParams({ days }, positionId) }),

  getScoreTrend: (days = 30, positionId = null) =>
    axiosClient.get('/hr/analytics/score-trend', { params: buildParams({ days }, positionId) }),

  getPositionsHealth: () =>
    axiosClient.get('/hr/analytics/positions-health'),
};
