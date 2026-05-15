import React from 'react';

const STAGE_CONFIG = {
  APPLIED:             { bg: '#EEF2FF', color: '#4338CA', label: 'Applied' },
  INTERVIEW_SCHEDULED: { bg: '#FFF7ED', color: '#C2410C', label: 'Interview Scheduled' },
  INTERVIEWED:         { bg: '#F3E8FF', color: '#7C3AED', label: 'Interviewed' },
  OFFER:               { bg: '#CFFAFE', color: '#0E7490', label: 'Offer' },
  ACCEPTED:            { bg: '#DCFCE7', color: '#16A34A', label: 'Accepted' },
  REJECTED:            { bg: '#FEE2E2', color: '#DC2626', label: 'Rejected' },
};

const StageTag = ({ stage }) => {
  const cfg = STAGE_CONFIG[stage];
  if (!cfg) return <span style={{ color: '#94A3B8', fontSize: 12 }}>{stage || '—'}</span>;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '3px 10px',
      borderRadius: 20,
      fontSize: 12,
      fontWeight: 600,
      background: cfg.bg,
      color: cfg.color,
      whiteSpace: 'nowrap',
      letterSpacing: '0.01em',
    }}>
      {cfg.label}
    </span>
  );
};

export default StageTag;
