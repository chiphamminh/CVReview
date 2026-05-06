import React from 'react';
import { Badge } from 'antd';

const ScoreBadge = ({ score }) => {
  if (score === null || score === undefined) {
    return <span style={{ color: '#8c8c8c' }}>Chưa chấm</span>;
  }

  let color = '#52c41a'; // Green
  if (score < 70) {
    color = '#f5222d'; // Red
  } else if (score < 80) {
    color = '#faad14'; // Warning/Orange
  }

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
      <Badge color={color} />
      <span style={{ fontWeight: 600, color }}>{score}/100</span>
    </div>
  );
};

export default ScoreBadge;
