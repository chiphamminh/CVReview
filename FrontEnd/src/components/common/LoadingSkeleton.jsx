import React from 'react';
import { Skeleton } from 'antd';

const LoadingSkeleton = ({ rows = 3, active = true, avatar = false }) => {
  return (
    <div style={{ padding: '16px' }}>
      <Skeleton active={active} avatar={avatar} paragraph={{ rows }} />
    </div>
  );
};

export default LoadingSkeleton;
