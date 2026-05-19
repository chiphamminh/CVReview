import React, { useState, useEffect } from 'react';
import { Button, Space, Input, Select, Typography, Dropdown, Tooltip, message, Modal, Tag, Badge, Segmented, Checkbox } from 'antd';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { SearchOutlined, EyeOutlined, FileTextOutlined, DownOutlined, EditOutlined, FilterOutlined, ReloadOutlined, DeleteOutlined, WarningOutlined } from '@ant-design/icons';
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
import useCandidateStore from '@/store/candidateStore';

const { Title, Text, Paragraph } = Typography;

const CandidatesPage = () => {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  const {
    searchInput, setSearchInput,
    keyword, setKeyword,
    stageFilter, setStageFilter,
    positionFilter, setPositionFilter,
    typeFilter, setTypeFilter,
    isScoredFilter, setIsScoredFilter,
    scoreSort, setScoreSort,
    page, pageSize, setPagination,
    clearAllFilters
  } = useCandidateStore();

  const [activeTab, setActiveTab] = useState('candidates');
  const [selectedBatchIds, setSelectedBatchIds] = useState([]);

  // Parse URL search params once on mount
  useEffect(() => {
    let changed = false;
    if (searchParams.has('positionId')) {
      setPositionFilter(parseInt(searchParams.get('positionId')));
      changed = true;
    }
    if (searchParams.has('sourceType')) {
      setTypeFilter(searchParams.get('sourceType'));
      changed = true;
    }
    if (changed) {
      setSearchParams({});
    }
  }, [searchParams, setPositionFilter, setTypeFilter, setSearchParams]);

  const [analysisModal, setAnalysisModal] = useState({ open: false, data: null });
  const [updateModal, setUpdateModal] = useState({ open: false, data: null });
  const [scheduleModal, setScheduleModal] = useState({ open: false, data: null, isReschedule: false });
  const [offerModal, setOfferModal] = useState({ open: false, data: null });

  // Debounce: chỉ cập nhật keyword (và reset page) sau 1000ms người dùng ngừng gõ
  useEffect(() => {
    const timer = setTimeout(() => {
      setKeyword(searchInput);
    }, 1000);
    return () => clearTimeout(timer);
  }, [searchInput, setKeyword]);

  const { data: candidatePage, isLoading: isCandLoading, isFetching: isCandFetching, refetch: refetchCandidates } = useQuery({
    queryKey: ['candidates', { keyword, stageFilter, positionFilter, typeFilter, isScoredFilter, scoreSort, page, pageSize }],
    queryFn: () => candidateApi.filter({
      keyword: keyword || undefined,
      positionId: positionFilter || undefined,
      stage: stageFilter || undefined,
      sourceType: typeFilter || undefined,
      isScored: isScoredFilter !== null ? isScoredFilter : undefined,
      scoreSort: scoreSort || undefined,
      page,
      size: pageSize,
    }),
    enabled: activeTab === 'candidates',
  });

  const { data: failedBatchesData, isFetching: isFailedFetching, refetch: refetchFailed } = useQuery({
    queryKey: ['failed-batches'],
    queryFn: () => candidateApi.getFailedBatches(),
    enabled: activeTab === 'failed',
  });

  const { data: positionsPage } = useQuery({
    queryKey: ['positions-dropdown'],
    queryFn: () => positionApi.filter({ size: 100 }),
  });

  const candidates = candidatePage?.data?.content ?? [];
  const totalElements = candidatePage?.data?.totalElements ?? 0;
  const positions = positionsPage?.data?.content ?? [];
  const failedBatches = failedBatchesData?.data ?? [];
  const filteredJobTitle = positionFilter ? positions.find(p => p.id === positionFilter)?.seniority + " " + positions.find(p => p.id === positionFilter)?.title : null;

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

  const deleteFailedMutation = useMutation({
    mutationFn: (batchIds) => candidateApi.deleteFailedBatches(batchIds),
    onSuccess: () => {
      message.success('Failed CVs deleted successfully.');
      setSelectedBatchIds([]);
      queryClient.invalidateQueries({ queryKey: ['failed-batches'] });
    },
    onError: () => message.error('Failed to delete CVs.'),
  });

  const handleClearAllFilters = () => {
    clearAllFilters();
    setSearchParams({});
  };

  const handleDeleteSelected = () => {
    Modal.confirm({
      title: `Delete ${selectedBatchIds.length} failed batch${selectedBatchIds.length > 1 ? 'es' : ''}?`,
      icon: <WarningOutlined style={{ color: '#ff4d4f' }} />,
      content: 'All CV files in the selected batches will be permanently deleted. This cannot be undone.',
      okText: 'Delete',
      okButtonProps: { danger: true },
      cancelText: 'Cancel',
      onOk: () => deleteFailedMutation.mutate(selectedBatchIds),
    });
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
        <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
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
          </Space>
          {getActionMenuItems(record).length > 0 && (
            <div style={{ marginLeft: 'auto' }}>
              <Dropdown menu={{ items: getActionMenuItems(record) }} trigger={['click']} placement="bottomRight">
                <Button type="link" size="small">
                  Action <DownOutlined />
                </Button>
              </Dropdown>
            </div>
          )}
        </div>
      ),
    },
  ];

  const cvDetailColumns = [
    {
      title: 'File Name',
      dataIndex: 'fileName',
      key: 'fileName',
      render: (name) => name
        ? <Text strong style={{ fontSize: 13 }}>{name}</Text>
        : <Text type="secondary" style={{ fontSize: 13 }}>Unknown file</Text>,
    },
    {
      title: 'Candidate',
      key: 'candidate',
      render: (_, record) => (
        record.name || record.email
          ? <div>
              {record.name && <div style={{ fontWeight: 500, fontSize: 13 }}>{record.name}</div>}
              {record.email && <Text type="secondary" style={{ fontSize: 12 }}>{record.email}</Text>}
            </div>
          : <Text type="secondary" style={{ fontSize: 13 }}>-</Text>
      ),
    },
    {
      title: 'Error',
      dataIndex: 'errorMessage',
      key: 'errorMessage',
      render: (msg) => msg
        ? <Paragraph ellipsis={{ rows: 2, tooltip: msg }} style={{ margin: 0, fontSize: 14, color: '#cf1322' }}>{msg}</Paragraph>
        : <Text type="secondary">-</Text>,
    },
    {
      title: 'View File',
      key: 'view',
      width: 80,
      render: (_, record) => record.driveFileUrl
        ? <Tooltip title="Open original file"><Button type="link" size="small" icon={<EyeOutlined />} onClick={() => window.open(record.driveFileUrl, '_blank')} /></Tooltip>
        : <Text type="secondary">-</Text>,
    },
  ];

  const failedColumns = [
    {
      title: '',
      key: 'checkbox',
      width: 40,
      render: (_, record) => (
        <Checkbox
          checked={selectedBatchIds.includes(record.batchId)}
          onChange={(e) => {
            setSelectedBatchIds(prev =>
              e.target.checked
                ? [...prev, record.batchId]
                : prev.filter(id => id !== record.batchId)
            );
          }}
        />
      ),
    },
    {
      title: 'Position',
      dataIndex: 'positionTitle',
      key: 'positionTitle',
      render: (title) => title || <Text type="secondary">-</Text>,
    },
    {
      title: 'Failed CVs',
      dataIndex: 'failedCount',
      key: 'failedCount',
      width: 100,
      render: (count) => <Tag color="red" style={{ fontSize: 14 }}>{count} file{count !== 1 ? 's' : ''}</Tag>,
    },
    {
      title: 'Uploaded At',
      dataIndex: 'uploadedAt',
      key: 'uploadedAt',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY HH:mm') : '-',
    },
    {
      title: 'Failed At',
      dataIndex: 'failedAt',
      key: 'failedAt',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY HH:mm') : '-',
    },
    {
      title: 'Error',
      dataIndex: 'errorMessage',
      key: 'errorMessage',
      render: (msg) => msg
        ? <Paragraph ellipsis={{ rows: 2, tooltip: msg }} style={{ margin: 0, fontSize: 14, color: '#cf1322' }}>{msg}</Paragraph>
        : <Text type="secondary">-</Text>,
    },
    {
      title: 'Actions',
      key: 'action',
      fixed: 'right',
      width: 80,
      render: (_, record) => (
        <Tooltip title="Delete batch">
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            loading={deleteFailedMutation.isPending}
            onClick={() => {
              Modal.confirm({
                title: 'Delete this failed batch?',
                icon: <WarningOutlined style={{ color: '#ff4d4f' }} />,
                content: `${record.failedCount} CV file${record.failedCount !== 1 ? 's' : ''} will be permanently deleted.`,
                okText: 'Delete',
                okButtonProps: { danger: true },
                cancelText: 'Cancel',
                onOk: () => deleteFailedMutation.mutate([record.batchId]),
              });
            }}
          />
        </Tooltip>
      ),
    },
  ];

  if (isCandLoading && candidates.length === 0 && activeTab === 'candidates') {
    return <LoadingSkeleton rows={12} />;
  }

  const allSelected = failedBatches.length > 0 && selectedBatchIds.length === failedBatches.length;
  const someSelected = selectedBatchIds.length > 0 && !allSelected;

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Title level={4} style={{ margin: 0 }}>
          Candidates Management
          {activeTab === 'candidates' && filteredJobTitle && (
            <span style={{ color: '#1677ff', marginLeft: 8 }}>- {filteredJobTitle}</span>
          )}
        </Title>
        <Segmented
          value={activeTab}
          onChange={(val) => {
            setActiveTab(val);
            setSelectedBatchIds([]);
          }}
          options={[
            { label: 'Candidates', value: 'candidates' },
            {
              label: (
                <Space size={6}>
                  <span>Failed CVs</span>
                  <Badge count={failedBatches.length} style={{ backgroundColor: failedBatches.length > 0 ? '#ff4d4f' : '#d9d9d9' }} />
                </Space>
              ),
              value: 'failed',
            },
          ]}
        />
      </div>

      {activeTab === 'candidates' && (
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
            placeholder="Filter by Position"
            style={{ width: 250 }}
            value={positionFilter}
            onChange={val => setPositionFilter(val ?? null)}
            options={positions.map(p => ({ value: p.id, label: p.seniority ? `${p.seniority} ${p.title}` : p.title }))}
          />
          <Select
            allowClear
            placeholder="Filter by Source Type"
            style={{ width: 150 }}
            value={typeFilter}
            onChange={val => setTypeFilter(val ?? null)}
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
            onChange={val => setStageFilter(val ?? null)}
            options={[
              { value: 'APPLIED', label: 'Applied' },
              { value: 'INTERVIEW_SCHEDULED', label: 'Interview Scheduled' },
              { value: 'INTERVIEWED', label: 'Interviewed' },
              { value: 'OFFER', label: 'Offer Extended' },
              { value: 'ACCEPTED', label: 'Accepted' },
              { value: 'REJECTED', label: 'Rejected' },
            ]}
          />
          <Select
            allowClear
            placeholder="Scoring Status"
            style={{ width: 140 }}
            value={isScoredFilter}
            onChange={val => setIsScoredFilter(val ?? null)}
            options={[
              { value: true, label: 'Scored' },
              { value: false, label: 'Not Scored' },
            ]}
          />
          <Select
            allowClear
            placeholder="Sort by"
            style={{ width: 160 }}
            value={scoreSort}
            onChange={val => setScoreSort(val ?? null)}
            options={[
              { value: 'desc', label: 'Highest Score' },
              { value: 'asc', label: 'Lowest Score' },
            ]}
          />
          <Button icon={<FilterOutlined />} onClick={handleClearAllFilters} danger type="dashed">
            Clear All
          </Button>
          <Button icon={<ReloadOutlined />} onClick={() => refetchCandidates()} loading={isCandFetching} style={{ marginLeft: 'auto' }}></Button>
        </div>
      )}

      {activeTab === 'failed' && (
        <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 12, background: '#fff2f0', padding: '10px 16px', borderRadius: 8, border: '1px solid #ffccc7' }}>
          <Checkbox
            indeterminate={someSelected}
            checked={allSelected}
            onChange={(e) => setSelectedBatchIds(e.target.checked ? failedBatches.map(b => b.batchId) : [])}
          >
            Select all
          </Checkbox>
          <Button
            danger
            icon={<DeleteOutlined />}
            disabled={selectedBatchIds.length === 0}
            loading={deleteFailedMutation.isPending}
            onClick={handleDeleteSelected}
          >
            Delete Selected ({selectedBatchIds.length})
          </Button>
          <Button icon={<ReloadOutlined />} onClick={() => refetchFailed()} loading={isFailedFetching} style={{ marginLeft: 'auto' }}>
            Refresh
          </Button>
        </div>
      )}

      {activeTab === 'candidates' && (
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
            setPagination(pagination.current - 1, pagination.pageSize);
          }}
        />
      )}

      {activeTab === 'failed' && (
        <AppTable
          columns={failedColumns}
          dataSource={failedBatches}
          rowKey="batchId"
          loading={isFailedFetching || deleteFailedMutation.isPending}
          pagination={false}
          expandable={{
            expandedRowRender: (record) => (
              <AppTable
                columns={cvDetailColumns}
                dataSource={record.cvs ?? []}
                rowKey="cvId"
                size="small"
                pagination={false}
                style={{ margin: '0 40px' }}
              />
            ),
            rowExpandable: (record) => record.cvs?.length > 0,
          }}
        />
      )}

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
