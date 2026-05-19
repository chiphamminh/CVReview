import { Typography } from 'antd';
import { Bar } from '@ant-design/plots';

const { Text } = Typography;

const STAGE_COLORS = ['#1677ff', '#722ed1', '#13c2c2', '#faad14', '#52c41a', '#ff4d4f'];

const PipelineOverviewChart = ({ data, isAllPositions, loading }) => {
  const chartData = [
    { stage: 'Applied', count: data?.applied ?? 0 },
    { stage: 'Interview Scheduled', count: data?.interviewScheduled ?? 0 },
    { stage: 'Interviewed', count: data?.interviewed ?? 0 },
    { stage: 'Offer', count: data?.offer ?? 0 },
    { stage: 'Accepted', count: data?.accepted ?? 0 },
    { stage: 'Rejected', count: data?.rejected ?? 0 },
  ];

  const total = chartData.reduce((s, d) => s + d.count, 0);

  const config = {
    data: chartData,
    xField: 'count',
    yField: 'stage',
    colorField: 'stage',
    scale: { color: { range: STAGE_COLORS } },
    label: {
      text: (d) => d.count > 0 ? d.count.toLocaleString() : '',
      position: 'right',
      style: { fill: '#595959', fontSize: 12 },
    },
    axis: {
      x: { title: 'Số CV', titleFill: '#8c8c8c', tickFilter: (t) => Number.isInteger(t) },
      y: { title: false },
    },
    tooltip: { items: [{ field: 'stage', name: 'Stage' }, { field: 'count', name: 'CVs' }] },
    autoFit: true,
  };

  if (!loading && total === 0) {
    return (
      <div style={{ height: 260, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Text type="secondary">Chưa có dữ liệu pipeline</Text>
      </div>
    );
  }

  return (
    <div>
      {isAllPositions && (
        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>
          Dữ liệu tổng hợp — chọn một vị trí cụ thể để xem pipeline chính xác hơn
        </Text>
      )}
      <Bar {...config} height={260} />
    </div>
  );
};

export default PipelineOverviewChart;
