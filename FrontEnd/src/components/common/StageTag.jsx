import React from 'react';
import { Tag } from 'antd';

const STAGE_COLORS = {
  'APPLIED': 'blue',
  'INTERVIEW_SCHEDULED': 'orange',
  'INTERVIEWED': 'cyan',
  'OFFER': 'gold',
  'ACCEPTED': 'green',
  'REJECTED': 'red',
};

const STAGE_LABELS = {
  'APPLIED': 'Đã nộp đơn',
  'INTERVIEW_SCHEDULED': 'Đã lên lịch phỏng vấn',
  'INTERVIEWED': 'Đã phỏng vấn',
  'OFFER': 'Đã gửi Offer',
  'ACCEPTED': 'Chấp nhận Offer',
  'REJECTED': 'Từ chối',
};

const StageTag = ({ stage }) => {
  const color = STAGE_COLORS[stage] || 'default';
  const label = STAGE_LABELS[stage] || stage;

  return (
    <Tag color={color} style={{ fontWeight: 500 }}>
      {label}
    </Tag>
  );
};

export default StageTag;
