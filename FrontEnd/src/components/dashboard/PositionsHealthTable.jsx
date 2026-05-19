import { Table, Tag, Typography } from 'antd';

const { Text } = Typography;

const HEALTH_CONFIG = {
  HEALTHY: { color: 'success', label: 'Healthy' },
  MEDIUM:  { color: 'warning', label: 'Medium' },
  LOW:     { color: 'error',   label: 'Low' },
};

const columns = [
  {
    title: 'Vị trí',
    dataIndex: 'title',
    key: 'title',
    render: (title, record) => (
      <div>
        <Text strong>{title}</Text>
        {record.seniority && <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>{record.seniority}</Text>}
      </div>
    ),
  },
  {
    title: 'Ngưỡng',
    dataIndex: 'minimumFitScore',
    key: 'minimumFitScore',
    width: 80,
    align: 'center',
    render: (v) => <Text>{v}</Text>,
  },
  {
    title: 'Pool Size',
    dataIndex: 'poolSize',
    key: 'poolSize',
    width: 90,
    align: 'center',
    sorter: (a, b) => a.poolSize - b.poolSize,
    render: (v) => v.toLocaleString(),
  },
  {
    title: 'Avg Score',
    dataIndex: 'avgScore',
    key: 'avgScore',
    width: 100,
    align: 'center',
    sorter: (a, b) => a.avgScore - b.avgScore,
    render: (v) => v > 0 ? <Text style={{ color: v >= 70 ? '#52c41a' : v >= 50 ? '#faad14' : '#ff4d4f' }}>{v}</Text> : '—',
  },
  {
    title: 'Qualified',
    dataIndex: 'qualifiedCount',
    key: 'qualifiedCount',
    width: 90,
    align: 'center',
    sorter: (a, b) => a.qualifiedCount - b.qualifiedCount,
    render: (v) => <Text strong>{v.toLocaleString()}</Text>,
  },
  {
    title: 'Health',
    dataIndex: 'healthStatus',
    key: 'healthStatus',
    width: 90,
    align: 'center',
    filters: [
      { text: 'Healthy', value: 'HEALTHY' },
      { text: 'Medium',  value: 'MEDIUM' },
      { text: 'Low',     value: 'LOW' },
    ],
    onFilter: (value, record) => record.healthStatus === value,
    render: (status) => {
      const cfg = HEALTH_CONFIG[status] ?? { color: 'default', label: status };
      return <Tag color={cfg.color}>{cfg.label}</Tag>;
    },
  },
];

const PositionsHealthTable = ({ data, loading }) => (
  <Table
    rowKey="id"
    loading={loading}
    columns={columns}
    dataSource={data?.positions ?? []}
    pagination={false}
    size="small"
    scroll={{ x: 600 }}
    locale={{ emptyText: 'Không có vị trí đang active' }}
  />
);

export default PositionsHealthTable;
