import { useState } from 'react';
import { Row, Col, Card, Typography, Segmented, Button, Space, Alert } from 'antd';
import {
  FileDoneOutlined, StarOutlined, ClockCircleOutlined,
  CheckCircleOutlined, ReloadOutlined,
} from '@ant-design/icons';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Column, Pie, Funnel } from '@ant-design/plots';
import KPICard from '@/components/dashboard/KPICard';
import { analyticsApi } from '@/api/analytics.api';

const { Text, Title } = Typography;

const SCORE_COLORS = ['#ff4d4f', '#fa8c16', '#fadb14', '#52c41a', '#1677ff'];
const STATUS_COLORS = ['#52c41a', '#1677ff', '#ff4d4f'];
const FUNNEL_COLORS = ['#1677ff', '#4096ff', '#69b1ff', '#91caff'];
const STALE_TIME = 60_000;

const HRDashboardPage = () => {
  const [days, setDays] = useState(30);
  const queryClient = useQueryClient();

  const {
    data: trafficRes, isLoading: trafficLoading, isError: trafficError,
  } = useQuery({
    queryKey: ['hr-traffic', days],
    queryFn: () => analyticsApi.getCvTraffic(days),
    staleTime: STALE_TIME,
    refetchInterval: STALE_TIME,
  });

  const {
    data: overviewRes, isLoading: overviewLoading, isError: overviewError,
  } = useQuery({
    queryKey: ['hr-overview', days],
    queryFn: () => analyticsApi.getOverview(days),
    staleTime: STALE_TIME,
    refetchInterval: STALE_TIME,
  });

  const {
    data: distributionRes, isLoading: distLoading, isError: distError,
  } = useQuery({
    queryKey: ['hr-score-distribution'],
    queryFn: () => analyticsApi.getScoreDistribution(),
    staleTime: 300_000,
  });

  const traffic = trafficRes?.data ?? {};
  const overview = overviewRes?.data ?? {};
  const buckets = distributionRes?.data?.buckets ?? [];

  const timeSavedHours = Math.ceil((traffic.successCv ?? 0) * 5 / 60);
  const isKpiLoading = trafficLoading || overviewLoading;
  const isKpiError = trafficError || overviewError;

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['hr-traffic'] });
    queryClient.invalidateQueries({ queryKey: ['hr-overview'] });
    queryClient.invalidateQueries({ queryKey: ['hr-score-distribution'] });
  };

  // ── Score Distribution chart config ──────────────────────────────────────
  const maxBucketCount = Math.max(...buckets.map((b) => b.count), 1);
  const columnConfig = {
    data: buckets,
    xField: 'range',
    yField: 'count',
    colorField: 'range',
    scale: {
      color: { range: SCORE_COLORS },
      // Force integer-only ticks on Y axis by restricting domain to whole numbers
      y: { domainMax: maxBucketCount, tickCount: Math.min(maxBucketCount + 1, 6), nice: false },
    },
    label: { text: 'count', position: 'top', style: { fill: '#595959', fontSize: 12 } },
    axis: {
      x: { title: 'Score Range', titleFill: '#8c8c8c' },
      y: {
        title: 'Number of CVs',
        titleFill: '#8c8c8c',
        // Filter out any non-integer ticks that G2 might still compute
        tickFilter: (tick) => Number.isInteger(tick),
      },
    },
    tooltip: { items: [{ field: 'label', name: 'Level' }, { field: 'count', name: 'CVs' }] },
    autoFit: true,
  };

  // ── CV Status Breakdown donut config ─────────────────────────────────────
  const totalCvs = traffic.totalCv ?? 0;
  const statusData = [
    { type: 'Processed', value: traffic.successCv ?? 0 },
    { type: 'Processing', value: traffic.processingCv ?? 0 },
    { type: 'Failed', value: traffic.failedCv ?? 0 },
  ].filter((d) => d.value > 0);

  const donutConfig = {
    data: statusData,
    angleField: 'value',
    colorField: 'type',
    innerRadius: 0.65,
    scale: { color: { range: STATUS_COLORS } },
    label: false,
    legend: { color: { position: 'right', layout: { justifyContent: 'center' } } },
    tooltip: { items: [{ field: 'type', name: 'Status' }, { field: 'value', name: 'Count' }] },
    // Center annotation showing total
    annotations: [
      {
        type: 'text',
        style: {
          x: '50%',
          y: '50%',
          text: `Total\n${totalCvs.toLocaleString()}`,
          textAlign: 'center',
          fontSize: 14,
          fontWeight: 600,
          fill: '#262626',
        },
      },
    ],
    autoFit: true,
  };

  // ── Processing Funnel config ──────────────────────────────────────────────
  const funnelData = [
    { stage: 'Uploaded', count: traffic.totalCv ?? 0 },
    { stage: 'Parsed', count: traffic.successCv ?? 0 },
    { stage: 'AI Scored', count: overview.totalCvsScored ?? 0 },
    { stage: 'Shortlisted', count: overview.totalCvsPassed ?? 0 },
  ];

  const funnelConfig = {
    data: funnelData,
    xField: 'stage',
    yField: 'count',
    colorField: 'stage',
    scale: { color: { range: FUNNEL_COLORS } },
    label: {
      text: (d, idx) => {
        const prev = funnelData[idx - 1];
        const rate = prev && prev.count > 0
          ? ` (${Math.round((d.count / prev.count) * 100)}%)`
          : '';
        return `${d.stage}: ${(d.count ?? 0).toLocaleString()}${rate}`;
      },
      position: 'inside',
      style: { fill: '#fff', fontSize: 12, fontWeight: 600 },
    },
    tooltip: { items: [{ field: 'stage', name: 'Stage' }, { field: 'count', name: 'CVs' }] },
    autoFit: true,
  };

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={4} style={{ margin: 0 }}>HR Dashboard</Title>
        <Space>
          <Segmented
            value={days}
            onChange={setDays}
            options={[
              { label: '7 Days', value: 7 },
              { label: '30 Days', value: 30 },
              { label: '90 Days', value: 90 },
            ]}
          />
          <Button icon={<ReloadOutlined />} onClick={handleRefresh}>Refresh</Button>
        </Space>
      </div>

      {/* KPI Cards — display:flex on each Col so all cards stretch to the same height */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} xl={6} style={{ display: 'flex' }}>
          <KPICard
            icon={<FileDoneOutlined />}
            iconColor="#1677ff"
            title="Total CVs Processed"
            value={isKpiError ? '—' : (traffic.successCv ?? 0)}
            loading={isKpiLoading}
          />
        </Col>
        <Col xs={24} sm={12} xl={6} style={{ display: 'flex' }}>
          <KPICard
            icon={<StarOutlined />}
            iconColor="#faad14"
            title="Avg Matching Score"
            value={isKpiError ? '—' : (overview.avgMatchingScore ?? 0)}
            suffix={isKpiError ? '' : '/ 100'}
            precision={1}
            loading={isKpiLoading}
          />
        </Col>
        <Col xs={24} sm={12} xl={6} style={{ display: 'flex' }}>
          <KPICard
            icon={<ClockCircleOutlined />}
            iconColor="#722ed1"
            title="Time Saved (Est.)"
            value={isKpiError ? '—' : timeSavedHours}
            suffix={isKpiError ? '' : 'hrs'}
            note="~5 min/CV"
            loading={isKpiLoading}
          />
        </Col>
        <Col xs={24} sm={12} xl={6} style={{ display: 'flex' }}>
          <KPICard
            icon={<CheckCircleOutlined />}
            iconColor="#52c41a"
            title="Success Match Rate"
            value={isKpiError ? '—' : (overview.successMatchRate ?? 0)}
            suffix={isKpiError ? '' : '%'}
            note="Score ≥ 70"
            precision={1}
            loading={isKpiLoading}
          />
        </Col>
      </Row>

      {/* Score Distribution + Processing Funnel */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={14}>
          <Card title="Candidate Score Distribution" loading={distLoading}>
            {distError ? (
              <ChartError />
            ) : buckets.length > 0 ? (
              <Column {...columnConfig} height={280} />
            ) : (
              <EmptyChart height={280} text="No scored CVs yet" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="Processing Funnel" loading={isKpiLoading}>
            {isKpiError ? (
              <ChartError />
            ) : funnelData[0].count > 0 ? (
              <Funnel {...funnelConfig} height={280} />
            ) : (
              <EmptyChart height={280} text="No CV data yet" />
            )}
          </Card>
        </Col>
      </Row>

      {/* CV Status Donut */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title={`CV Status Breakdown (Last ${days} Days)`} loading={trafficLoading}>
            {trafficError ? (
              <ChartError />
            ) : statusData.length > 0 ? (
              <Pie {...donutConfig} height={260} />
            ) : (
              <EmptyChart height={260} text="No data available" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

const EmptyChart = ({ height, text }) => (
  <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Text type="secondary">{text}</Text>
  </div>
);

const ChartError = () => (
  <Alert
    type="error"
    message="Failed to load data"
    description="Could not reach the analytics service. Please refresh or try again later."
    showIcon
    style={{ margin: '12px 0' }}
  />
);

export default HRDashboardPage;
