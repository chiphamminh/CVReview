import React from 'react';
import { Empty } from 'antd';

const EmptyState = ({ description = 'Không có dữ liệu', image = Empty.PRESENTED_IMAGE_SIMPLE }) => {
  return (
    <div style={{ padding: '32px 0' }}>
      <Empty image={image} description={description} />
    </div>
  );
};

export default EmptyState;
