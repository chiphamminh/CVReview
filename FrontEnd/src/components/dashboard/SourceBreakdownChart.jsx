import { Row, Col, Statistic, Divider, Typography, Empty } from 'antd';
import { TeamOutlined, UserOutlined } from '@ant-design/icons';

const { Text } = Typography;

const MetricRow = ({ label, internal, external, suffix = '' }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid #f0f0f0' }}>
    <Text type="secondary" style={{ fontSize: 13, flex: 1 }}>{label}</Text>
    <Text strong style={{ width: 80, textAlign: 'center' }}>{internal}{suffix}</Text>
    <Text strong style={{ width: 80, textAlign: 'center' }}>{external}{suffix}</Text>
  </div>
);

const SourceBreakdownChart = ({ data }) => {
  const hasData = (data?.internalCount ?? 0) + (data?.externalCount ?? 0) > 0;

  if (!hasData) {
    return <Empty description="Chưa có dữ liệu nguồn CV" image={Empty.PRESENTED_IMAGE_SIMPLE} style={{ margin: '24px 0' }} />;
  }

  const internalPassRate = data.internalScored > 0
    ? ((data.internalPassed / data.internalScored) * 100).toFixed(1) : '—';
  const externalPassRate = data.externalScored > 0
    ? ((data.externalPassed / data.externalScored) * 100).toFixed(1) : '—';

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <Text type="secondary" style={{ flex: 1 }} />
        <div style={{ width: 80, textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
          <UserOutlined style={{ color: '#1677ff' }} />
          <Text strong style={{ color: '#1677ff', fontSize: 12 }}>INTERNAL</Text>
        </div>
        <div style={{ width: 80, textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
          <TeamOutlined style={{ color: '#52c41a' }} />
          <Text strong style={{ color: '#52c41a', fontSize: 12 }}>EXTERNAL</Text>
        </div>
      </div>

      <MetricRow label="Tổng CV" internal={(data.internalCount ?? 0).toLocaleString()} external={(data.externalCount ?? 0).toLocaleString()} />
      <MetricRow label="Đã chấm điểm" internal={(data.internalScored ?? 0).toLocaleString()} external={(data.externalScored ?? 0).toLocaleString()} />
      <MetricRow label="Đạt ngưỡng" internal={(data.internalPassed ?? 0).toLocaleString()} external={(data.externalPassed ?? 0).toLocaleString()} />
      <MetricRow label="Pass rate" internal={`${internalPassRate}%`} external={`${externalPassRate}%`} />
      <MetricRow label="Avg score" internal={data.internalAvgScore > 0 ? data.internalAvgScore : '—'} external={data.externalAvgScore > 0 ? data.externalAvgScore : '—'} suffix={data.internalAvgScore > 0 || data.externalAvgScore > 0 ? '' : ''} />
    </div>
  );
};

export default SourceBreakdownChart;
