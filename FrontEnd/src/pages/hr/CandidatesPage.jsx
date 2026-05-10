import React, { useState, useEffect } from 'react';
import { Button, Space, Input, Select, Typography, Dropdown, Tooltip, message, Modal, Tag } from 'antd';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { SearchOutlined, EyeOutlined, FileTextOutlined, DownOutlined, EditOutlined, FilterOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { useSearchParams } from 'react-router-dom';

import AppTable from '@/components/tables/AppTable';
import LoadingSkeleton from '@/components/common/LoadingSkeleton';
import StageTag from '@/components/common/StageTag';
import ScoreBadge from '@/components/common/ScoreBadge';
import UpdateCVModal from '@/components/modals/UpdateCVModal';
import ScheduleInterviewModal from '@/components/modals/ScheduleInterviewModal';
import SendOfferModal from '@/components/modals/SendOfferModal';
import { candidateApi } from '@/api/candidate.api';
import { positionApi } from '@/api/position.api';

const { Title, Text, Paragraph } = Typography;

const CandidatesPage = () => {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  const positionIdParam = searchParams.get('positionId') ? parseInt(searchParams.get('positionId')) : null;
  const sourceTypeParam = searchParams.get('sourceType') || null;

  // searchInput: giá trị hiển thị trong ô input (cập nhật ngay)
  // keyword: giá trị debounced 500ms → dùng để trigger API call
  const [searchInput, setSearchInput] = useState('');
  const [keyword, setKeyword] = useState('');
  const [stageFilter, setStageFilter] = useState(null);
  const [positionFilter, setPositionFilter] = useState(positionIdParam);
  const [typeFilter, setTypeFilter] = useState(sourceTypeParam);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);

  const [analysisModal, setAnalysisModal] = useState({ open: false, data: null });
  const [updateModal, setUpdateModal] = useState({ open: false, data: null });
  const [scheduleModal, setScheduleModal] = useState({ open: false, data: null, isReschedule: false });
  const [offerModal, setOfferModal] = useState({ open: false, data: null });

  // Debounce: chỉ cập nhật keyword (và reset page) sau 1000ms người dùng ngừng gõ
  useEffect(() => {
    const timer = setTimeout(() => {
      setKeyword(searchInput);
      setPage(0);
    }, 1000);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const { data: candidatePage, isLoading: isCandLoading } = useQuery({
    queryKey: ['candidates', { keyword, stageFilter, positionFilter, typeFilter, page, pageSize }],
    queryFn: () => candidateApi.filter({
      keyword: keyword || undefined,
      positionId: positionFilter || undefined,
      stage: stageFilter || undefined,
      sourceType: typeFilter || undefined,
      page,
      size: pageSize,
    }),
  });

  const { data: positionsPage } = useQuery({
    queryKey: ['positions-dropdown'],
    queryFn: () => positionApi.filter({ size: 100 }),
  });

  const candidates = candidatePage?.data?.content ?? [];
  const totalElements = candidatePage?.data?.totalElements ?? 0;
  const positions = positionsPage?.data?.content ?? [];
  const filteredJobTitle = positionFilter ? positions.find(p => p.id === positionFilter)?.title : null;

  const updateStageMutation = useMutation({
    mutationFn: ({ cvId, stage }) => candidateApi.updateStage(cvId, stage),
    onSuccess: () => {
      message.success('Candidate stage updated successfully!');
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
    },
    onError: () => message.error('Failed to update stage.'),
  });

  const scheduleMutation = useMutation({
    mutationFn: ({ cvId, payload, isReschedule }) => isReschedule
      ? candidateApi.rescheduleInterview(cvId, payload)
      : candidateApi.scheduleInterview(cvId, payload),
    onSuccess: (_, vars) => {
      message.success(vars.isReschedule ? 'Interview rescheduled!' : 'Interview scheduled and invitation sent!');
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
      setScheduleModal({ open: false, data: null, isReschedule: false });
    },
    onError: () => message.error('Failed to schedule interview.'),
  });

  const offerMutation = useMutation({
    mutationFn: ({ cvId, payload }) => candidateApi.sendOffer(cvId, payload),
    onSuccess: () => {
      message.success('Offer sent successfully!');
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
      setOfferModal({ open: false, data: null });
    },
    onError: () => message.error('Failed to send offer.'),
  });

  const updateCVMutation = useMutation({
    mutationFn: ({ cvId, params }) => candidateApi.updateInfo(cvId, params),
    onSuccess: () => {
      message.success('Candidate info updated!');
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
      setUpdateModal({ open: false, data: null });
    },
    onError: () => message.error('Failed to update candidate info.'),
  });

  const clearAllFilters = () => {
    setSearchInput('');
    setKeyword('');
    setStageFilter(null);
    setPositionFilter(null);
    setTypeFilter(null);
    setPage(0);
    setSearchParams({});
  };

  const getActionMenuItems = (record) => {
    const stage = record.recruitmentStage;
    const items = [];

    if (stage === 'APPLIED') {
      items.push({ key: 'SCHEDULE_INTERVIEW', label: 'Schedule Interview' });
      items.push({ key: 'REJECTED', label: 'Reject', danger: true });
    } else if (stage === 'INTERVIEW_SCHEDULED') {
      items.push({ key: 'RE_SCHEDULE', label: 'Re-schedule Interview' });
      items.push({ key: 'REJECTED', label: 'Reject', danger: true });
    } else if (stage === 'INTERVIEWED') {
      items.push({ key: 'OFFER', label: 'Send Offer' });
      items.push({ key: 'REJECTED', label: 'Reject', danger: true });
    } else if (stage === 'OFFER') {
      items.push({ key: 'ACCEPTED', label: 'Candidate Accepted' });
      items.push({ key: 'REJECTED', label: 'Candidate Rejected', danger: true });
    }

    return items.map(item => ({
      ...item,
      onClick: () => {
        if (item.key === 'SCHEDULE_INTERVIEW') {
          setScheduleModal({ open: true, data: record, isReschedule: false });
        } else if (item.key === 'RE_SCHEDULE') {
          setScheduleModal({ open: true, data: record, isReschedule: true });
        } else if (item.key === 'OFFER') {
          setOfferModal({ open: true, data: record });
        } else {
          updateStageMutation.mutate({ cvId: record.cvId, stage: item.key });
        }
      },
    }));
  };

  const columns = [
    {
      title: 'Candidate Name',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.email}</Text>
        </div>
      ),
    },
    {
      title: 'Job Title',
      dataIndex: 'positionTitle',
      key: 'positionTitle',
      render: (title) => title || '-',
    },
    {
      title: 'Type',
      dataIndex: 'sourceType',
      key: 'sourceType',
      render: (type) => (
        <Tag color={type === 'INTERNAL' ? 'geekblue' : 'green'}>{type}</Tag>
      ),
    },
    {
      title: 'Applied Date',
      dataIndex: 'appliedDate',
      key: 'appliedDate',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY') : '-',
    },
    {
      title: 'AI Score',
      key: 'score',
      render: (_, record) => {
        const score = record.technicalScore != null && record.experienceScore != null
          ? Math.round((record.technicalScore + record.experienceScore) / 2)
          : null;
        return <ScoreBadge score={score} />;
      },
    },
    {
      title: 'Reason for Match',
      key: 'reason',
      width: 250,
      render: (_, record) => (
        <Paragraph ellipsis={{ rows: 2, tooltip: record.aiAssessment }} style={{ margin: 0, fontSize: 13, color: '#595959' }}>
          {record.aiAssessment || '-'}
        </Paragraph>
      ),
    },
    {
      title: 'Stage',
      dataIndex: 'recruitmentStage',
      key: 'recruitmentStage',
      render: (stage) => <StageTag stage={stage} />,
    },
    {
      title: 'Interview Schedule',
      dataIndex: 'interviewSchedule',
      key: 'interviewSchedule',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY HH:mm') : <Text type="secondary">-</Text>,
    },
    {
      title: 'Actions',
      key: 'action',
      fixed: 'right',
      width: 180,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="View CV">
            <Button type="text" icon={<EyeOutlined />} onClick={() => window.open(record.driveFileUrl, '_blank')} />
          </Tooltip>
          <Tooltip title="AI Analysis">
            <Button type="text" icon={<FileTextOutlined />} onClick={() => setAnalysisModal({ open: true, data: record })} />
          </Tooltip>
          {record.sourceType === 'INTERNAL' && (
            <Tooltip title="Edit Info">
              <Button type="text" icon={<EditOutlined />} onClick={() => setUpdateModal({ open: true, data: record })} />
            </Tooltip>
          )}
          {getActionMenuItems(record).length > 0 && (
            <Dropdown menu={{ items: getActionMenuItems(record) }} trigger={['click']}>
              <Button type="link" size="small">
                Action <DownOutlined />
              </Button>
            </Dropdown>
          )}
        </Space>
      ),
    },
  ];

  if (isCandLoading && candidates.length === 0) {
    return <LoadingSkeleton rows={12} />;
  }

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>
          Candidates Management
          {filteredJobTitle && <span style={{ color: '#1677ff', marginLeft: 8 }}>- {filteredJobTitle}</span>}
        </Title>
      </div>

      <div style={{ marginBottom: 16, display: 'flex', flexWrap: 'wrap', gap: 16, background: '#fafafa', padding: 16, borderRadius: 8 }}>
        <Input
          placeholder="Search name or email..."
          prefix={<SearchOutlined />}
          style={{ width: 250 }}
          value={searchInput}
          onChange={e => setSearchInput(e.target.value)}
          allowClear
        />
        <Select
          allowClear
          showSearch
          optionFilterProp="label"
          placeholder="Filter by Position"
          style={{ width: 250 }}
          value={positionFilter}
          onChange={val => { setPositionFilter(val ?? null); setPage(0); }}
          options={positions.map(p => ({ value: p.id, label: p.title }))}
        />
        <Select
          allowClear
          placeholder="Filter by Source Type"
          style={{ width: 150 }}
          value={typeFilter}
          onChange={val => { setTypeFilter(val ?? null); setPage(0); }}
          options={[
            { value: 'INTERNAL', label: 'Internal' },
            { value: 'EXTERNAL', label: 'External' },
          ]}
        />
        <Select
          allowClear
          placeholder="Filter by Stage"
          style={{ width: 200 }}
          value={stageFilter}
          onChange={val => { setStageFilter(val ?? null); setPage(0); }}
          options={[
            { value: 'APPLIED', label: 'Applied' },
            { value: 'INTERVIEW_SCHEDULED', label: 'Interview Scheduled' },
            { value: 'INTERVIEWED', label: 'Interviewed' },
            { value: 'OFFER', label: 'Offer Extended' },
            { value: 'ACCEPTED', label: 'Accepted' },
            { value: 'REJECTED', label: 'Rejected' },
          ]}
        />
        <Button icon={<FilterOutlined />} onClick={clearAllFilters} danger type="dashed">
          Clear All
        </Button>
      </div>

      <AppTable
        columns={columns}
        dataSource={candidates}
        rowKey="cvId"
        loading={
          isCandLoading ||
          updateStageMutation.isPending ||
          scheduleMutation.isPending ||
          updateCVMutation.isPending ||
          offerMutation.isPending
        }
        pagination={{
          current: page + 1,
          pageSize,
          total: totalElements,
        }}
        onChange={(pagination) => {
          setPage(pagination.current - 1);
          setPageSize(pagination.pageSize);
        }}
      />

      <Modal
        title="AI Analysis Details"
        open={analysisModal.open}
        onCancel={() => setAnalysisModal({ open: false, data: null })}
        footer={[<Button key="close" onClick={() => setAnalysisModal({ open: false, data: null })}>Close</Button>]}
      >
        {analysisModal.data ? (
          <div>
            <p><strong>Technical Score:</strong> {analysisModal.data.technicalScore ?? 'N/A'}</p>
            <p><strong>Experience Score:</strong> {analysisModal.data.experienceScore ?? 'N/A'}</p>
            <p><strong>Reason for Match:</strong> {analysisModal.data.aiAssessment || '-'}</p>
            {analysisModal.data.learningPath && (
              <p><strong>Learning Path:</strong> {analysisModal.data.learningPath}</p>
            )}
          </div>
        ) : (
          <p>No analysis data available</p>
        )}
      </Modal>

      <UpdateCVModal
        open={updateModal.open}
        onCancel={() => setUpdateModal({ open: false, data: null })}
        initialData={updateModal.data}
        onSave={(data) => updateCVMutation.mutate({ cvId: updateModal.data.cvId, params: data })}
      />

      <ScheduleInterviewModal
        open={scheduleModal.open}
        onCancel={() => setScheduleModal({ open: false, data: null, isReschedule: false })}
        candidateData={scheduleModal.data}
        onSave={(data) => scheduleMutation.mutate({
          cvId: scheduleModal.data.cvId,
          payload: { interviewDate: data.date, customMessage: data.note },
          isReschedule: scheduleModal.isReschedule,
        })}
      />

      <SendOfferModal
        open={offerModal.open}
        onCancel={() => setOfferModal({ open: false, data: null })}
        candidateData={offerModal.data}
        loading={offerMutation.isPending}
        onSave={(data) => offerMutation.mutate({
          cvId: offerModal.data.cvId,
          payload: {
            startDate: data.startDate,
            offerExpirationDate: data.expirationDate,
            files: data.files,
          },
        })}
      />
    </div>
  );
};

export default CandidatesPage;
