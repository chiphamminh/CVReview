import React from 'react';

const ScoreBadge = ({ score }) => {
  if (score === null || score === undefined) {
    return <span style={{ color: '#94A3B8', fontSize: 13 }}>—</span>;
  }

  let color;
  if (score < 70)      color = '#DC2626';
  else if (score < 80) color = '#D97706';
  else                 color = '#16A34A';

  return (
    <div style={{ display: 'inline-flex', flexDirection: 'column', gap: 4, minWidth: 72 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
        <span style={{ fontWeight: 700, color, fontSize: 14, lineHeight: 1 }}>{score}</span>
        <span style={{ color: '#94A3B8', fontSize: 11 }}>/100</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: '#E2E8F0', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${score}%`,
          background: color,
          borderRadius: 2,
          transition: 'width 0.5s ease',
        }} />
      </div>
    </div>
  );
};

export default ScoreBadge;
