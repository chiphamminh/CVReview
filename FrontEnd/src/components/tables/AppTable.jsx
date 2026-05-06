import React from 'react';
import { Table } from 'antd';

const AppTable = ({ 
  columns, 
  dataSource, 
  loading, 
  rowKey = 'id', 
  pagination, 
  onChange,
  ...restProps 
}) => {
  return (
    <Table
      columns={columns}
      dataSource={dataSource}
      loading={loading}
      rowKey={rowKey}
      pagination={pagination ? {
        ...pagination,
        showSizeChanger: true,
        showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`,
      } : false}
      onChange={onChange}
      scroll={{ x: 'max-content' }}
      bordered
      size="middle"
      {...restProps}
    />
  );
};

export default AppTable;
