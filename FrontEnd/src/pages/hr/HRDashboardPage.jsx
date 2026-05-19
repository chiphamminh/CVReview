import { useState, useMemo } from 'react';
import { Row, Col, Card, Typography, Segmented, Button, Space, Alert, Select } from 'antd';
import {
  FileDoneOutlined, StarOutlined, ClockCircleOutlined,
  CheckCircleOutlined, ReloadOutlined, AimOutlined, TeamOutlined,
} from '@ant-design/icons';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Column, Pie } from '@ant-design/plots';
import KPICard from '@/components/dashboard/KPICard';
import PipelineOverviewChart from '@/components/dashboard/PipelineOverviewChart';
import SourceBreakdownChart from '@/components/dashboard/SourceBreakdownChart';
import ScoreTrendChart from '@/components/dashboard/ScoreTrendChart';
import PositionsHealthTable from '@/components/dashboard/PositionsHealthTable';
import { analyticsApi } from '@/api/analytics.api';

const { Text, Title } = Typography;

const SCORE_COLORS = ['#ff4d4f', '#fa8c16', '#fadb14', '#52c41a', '#1677ff'];
const STATUS_COLORS = ['#52c41a', '#1677ff', '#ff4d4f'];
const STALE_TIME = 60_000;

const THRESHOLD_HEALTH = (rate) => {
  if (rate < 5)  return { label: 'Too High',  color: '#ff4d4f',  icon: '🔴' };
  if (rate <= 30) return { label: 'Balanced',  color: '#52c41a',  icon: '🟢' };
  if (rate <= 60) return { label: 'Generous',  color: '#faad14',  icon: '🟡' };
  return            { label: 'Low Bar',    color: '#fa541c',  icon: '🟠' };
};

const HRDashboardPage = () => {
  const [days, setDays] = useState(30);
  const [selectedPositionId, setSelectedPositionId] = useState(null);
  const queryClient = useQueryClient();

  // ── Active positions for dropdown ──────────────────────────────────────────
  const { data: positionsRes } = useQuery({
    queryKey: ['active-positions'],
    queryFn: () => analyticsApi.getActivePositions(),
    staleTime: 300_000,
  });
  const positions = positionsRes?.data ?? [];
  const selectedPosition = useMemo(
    () => positions.find((p) => p.id === selectedPositionId) ?? null,
    [positions, selectedPositionId],
  );
  const threshold = selectedPosition?.minimumFitScore ?? 70;

  // ── Core KPI queries (adapt to position filter) ────────────────────────────
  const { data: trafficRes, isLoading: trafficLoading, isError: trafficError } = useQuery({
    queryKey: ['hr-traffic', days, selectedPositionId],
    queryFn: () => analyticsApi.getCvTraffic(days, selectedPositionId),
    staleTime: STALE_TIME,
  });

  const { data: overviewRes, isLoading: overviewLoading, isError: overviewError } = useQuery({
    queryKey: ['hr-overview', days, selectedPositionId],
    queryFn: () => analyticsApi.getOverview(days, selectedPositionId),
    staleTime: STALE_TIME,
  });

  const { data: distributionRes, isLoading: distLoading, isError: distError } = useQuery({
    queryKey: ['hr-score-distribution', selectedPositionId],
    queryFn: () => analyticsApi.getScoreDistribution(selectedPositionId),
    staleTime: 300_000,
  });

  // ── New feature queries ─────────────────────────────────────────────────────
  const { data: pipelineRes, isLoading: pipelineLoading, isError: pipelineError } = useQuery({
    queryKey: ['hr-stage-pipeline', selectedPositionId],
    queryFn: () => analyticsApi.getStagePipeline(selectedPositionId),
    staleTime: STALE_TIME,
  });

  const { data: sourceRes, isLoading: sourceLoading, isError: sourceError } = useQuery({
    queryKey: ['hr-source-breakdown', days, selectedPositionId],
    queryFn: () => analyticsApi.getSourceBreakdown(days, selectedPositionId),
    staleTime: STALE_TIME,
  });

  const { data: trendRes, isLoading: trendLoading, isError: trendError } = useQuery({
    queryKey: ['hr-score-trend', days, selectedPositionId],
    queryFn: () => analyticsApi.getScoreTrend(days, selectedPositionId),
    staleTime: STALE_TIME,
  });

  const { data: healthRes, isLoading: healthLoading } = useQuery({
    queryKey: ['hr-positions-health'],
    queryFn: () => analyticsApi.getPositionsHealth(),
    staleTime: 120_000,
  });

  // ── Derived values ──────────────────────────────────────────────────────────
  const traffic = trafficRes?.data ?? {};
  const overview = overviewRes?.data ?? {};
  const buckets = distributionRes?.data?.buckets ?? [];
  const pipeline = pipelineRes?.data ?? {};
  const source = sourceRes?.data ?? {};
  const trend = trendRes?.data ?? {};
  const health = healthRes?.data ?? {};

  const isKpiLoading = trafficLoading || overviewLoading;
  const isKpiError = trafficError || overviewError;
  const timeSavedHours = Math.ceil((traffic.successCv ?? 0) * 5 / 60);
  const thresholdHealth = THRESHOLD_HEALTH(overview.successMatchRate ?? 0);

  // Score distribution: sparsity guard when filtering a specific position
  const totalBucketCount = buckets.reduce((s, b) => s + b.count, 0);
  const showSparsityWarning = selectedPositionId !== null && totalBucketCount < 5;

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['hr-traffic'] });
    queryClient.invalidateQueries({ queryKey: ['hr-overview'] });
    queryClient.invalidateQueries({ queryKey: ['hr-score-distribution'] });
    queryClient.invalidateQueries({ queryKey: ['hr-stage-pipeline'] });
    queryClient.invalidateQueries({ queryKey: ['hr-source-breakdown'] });
    queryClient.invalidateQueries({ queryKey: ['hr-score-trend'] });
    queryClient.invalidateQueries({ queryKey: ['hr-positions-health'] });
  };

  // ── Score Distribution chart config ────────────────────────────────────────
  const maxBucketCount = Math.max(...buckets.map((b) => b.count), 1);
  const columnConfig = {
    data: buckets,
    xField: 'range',
    yField: 'count',
    colorField: 'range',
    scale: {
      color: { range: SCORE_COLORS },
      y: { domainMax: maxBucketCount, tickCount: Math.min(maxBucketCount + 1, 6), nice: false },
    },
    label: { text: 'count', position: 'top', style: { fill: '#595959', fontSize: 12 } },
    axis: {
      x: { title: 'Score Range', titleFill: '#8c8c8c' },
      y: {
        title: 'Number of CVs',
        titleFill: '#8c8c8c',
        tickFilter: (tick) => Number.isInteger(tick),
      },
    },
    tooltip: { items: [{ field: 'label', name: 'Level' }, { field: 'count', name: 'CVs' }] },
    autoFit: true,
  };

  // ── CV Status Breakdown donut config ───────────────────────────────────────
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
    annotations: [{
      type: 'text',
      style: {
        x: '50%', y: '50%',
        text: `Total\n${totalCvs.toLocaleString()}`,
        textAlign: 'center', fontSize: 14, fontWeight: 600, fill: '#262626',
      },
    }],
    autoFit: true,
  };

  // ── KPI cards definition ───────────────────────────────────────────────────
  const kpiCards = [
    {
      key: 'pool',
      icon: <FileDoneOutlined />,
      iconColor: '#1677ff',
      title: selectedPositionId ? 'Pool Size' : 'Total CVs Processed',
      value: isKpiError ? '—' : (selectedPositionId ? (traffic.totalCv ?? 0) : (traffic.successCv ?? 0)),
      loading: isKpiLoading,
    },
    {
      key: 'score',
      icon: <StarOutlined />,
      iconColor: '#faad14',
      title: 'Avg Matching Score',
      value: isKpiError ? '—' : (overview.avgMatchingScore ?? 0),
      suffix: isKpiError ? '' : '/ 100',
      precision: 1,
      loading: isKpiLoading,
    },
    {
      key: 'qualified',
      icon: <CheckCircleOutlined />,
      iconColor: '#52c41a',
      title: 'Pass Rate',
      value: isKpiError ? '—' : (overview.successMatchRate ?? 0),
      suffix: isKpiError ? '' : '%',
      note: `Score ≥ ${threshold}`,
      precision: 1,
      loading: isKpiLoading,
    },
    {
      key: 'time',
      icon: <ClockCircleOutlined />,
      iconColor: '#722ed1',
      title: 'Time Saved (Est.)',
      value: isKpiError ? '—' : timeSavedHours,
      suffix: isKpiError ? '' : 'hrs',
      note: '~5 min/CV',
      loading: isKpiLoading,
    },
    // Threshold Health card — only shown when a specific position is selected
    ...(selectedPositionId !== null ? [{
      key: 'threshold',
      icon: <AimOutlined />,
      iconColor: thresholdHealth.color,
      title: 'Threshold Health',
      value: isKpiError ? '—' : thresholdHealth.label,
      note: `Threshold: ${threshold}`,
      loading: isKpiLoading,
    }] : []),
  ];

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <Title level={4} style={{ margin: 0 }}>HR Dashboard</Title>
        <Space wrap>
          <Select
            style={{ width: 240 }}
            placeholder="All Positions"
            allowClear
            value={selectedPositionId}
            onChange={(val) => setSelectedPositionId(val ?? null)}
            options={positions.map((p) => ({
              value: p.id,
              label: `${p.title}${p.seniority ? ` — ${p.seniority}` : ''}`,
            }))}
          />
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

      {/* KPI Cards — flex container supports 4 or 5 cards without layout shift */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
        {kpiCards.map((card) => (
          <div key={card.key} style={{ flex: '1 1 180px', minWidth: 0 }}>
            <KPICard {...card} />
          </div>
        ))}
      </div>

      {/* Score Distribution + Pipeline Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={14}>
          <Card title="Candidate Score Distribution" loading={distLoading}>
            {distError ? (
              <ChartError />
            ) : showSparsityWarning ? (
              <EmptyChart height={280} text="Cần ít nhất 5 CV để hiển thị phân phối điểm" />
            ) : buckets.length > 0 ? (
              <Column {...columnConfig} height={280} />
            ) : (
              <EmptyChart height={280} text="No scored CVs yet" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="Recruitment Pipeline" loading={pipelineLoading}>
            {pipelineError ? (
              <ChartError />
            ) : (
              <PipelineOverviewChart
                data={pipeline}
                isAllPositions={selectedPositionId === null}
                loading={pipelineLoading}
              />
            )}
          </Card>
        </Col>
      </Row>

      {/* Source Breakdown + Score Trend */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={10}>
          <Card
            title={<Space><TeamOutlined />Source Quality</Space>}
            loading={sourceLoading}
            extra={<Text type="secondary" style={{ fontSize: 12 }}>{selectedPositionId ? 'All time' : `Last ${days} days`}</Text>}
          >
            {sourceError ? <ChartError /> : <SourceBreakdownChart data={source} />}
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card
            title="Score Trend"
            loading={trendLoading}
            extra={<Text type="secondary" style={{ fontSize: 12 }}>Last {days} days</Text>}
          >
            {trendError ? <ChartError /> : <ScoreTrendChart data={trend} />}
          </Card>
        </Col>
      </Row>

      {/* CV Status Breakdown */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
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

      {/* Positions Health Table */}
      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <Card title="Positions Health Overview">
            <PositionsHealthTable data={health} loading={healthLoading} />
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
