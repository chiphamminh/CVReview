import { Typography } from 'antd';
import { Line } from '@ant-design/plots';

const { Text } = Typography;

const ScoreTrendChart = ({ data }) => {
  const points = data?.points ?? [];

  if (points.length === 0) {
    return (
      <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Text type="secondary">Chưa đủ dữ liệu để hiển thị xu hướng điểm</Text>
      </div>
    );
  }

  // Single data point: Line chart won't render meaningfully, show a note
  if (points.length === 1) {
    return (
      <div style={{ height: 240, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
        <Text style={{ fontSize: 32, fontWeight: 700, color: '#1677ff' }}>{points[0].avgScore}</Text>
        <Text type="secondary">Avg score — {points[0].weekLabel} ({points[0].cvCount} CVs)</Text>
        <Text type="secondary" style={{ fontSize: 12 }}>Cần ít nhất 2 tuần dữ liệu để hiển thị đường xu hướng</Text>
      </div>
    );
  }

  const config = {
    data: points,
    xField: 'weekLabel',
    yField: 'avgScore',
    smooth: true,
    point: { shapeField: 'circle', sizeField: 4 },
    scale: { y: { domain: [0, 100], nice: false } },
    axis: {
      x: { title: 'Tuần', titleFill: '#8c8c8c' },
      y: { title: 'Avg Score', titleFill: '#8c8c8c', domain: [0, 100] },
    },
    style: { stroke: '#1677ff', lineWidth: 2 },
    tooltip: {
      items: [
        { field: 'weekLabel', name: 'Tuần' },
        { field: 'avgScore', name: 'Avg Score' },
        { field: 'cvCount', name: 'Số CVs' },
      ],
    },
    autoFit: true,
  };

  return <Line {...config} height={240} />;
};

export default ScoreTrendChart;
