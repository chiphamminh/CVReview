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
import { fetchCandidates, updateCandidateStage, fetchPositions, scheduleCandidateInterview, updateCandidateCVInfo } from '@/api/mockData';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const CandidatesPage = () => {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // URL Params State
  const positionIdParam = searchParams.get('positionId') ? parseInt(searchParams.get('positionId')) : null;
  const typeParam = searchParams.get('type') || null;

  // Local State Filters
  const [searchText, setSearchText] = useState('');
  const [stageFilter, setStageFilter] = useState(null);
  const [positionFilter, setPositionFilter] = useState(positionIdParam);
  const [typeFilter, setTypeFilter] = useState(typeParam);

  // Modals State
  const [analysisModal, setAnalysisModal] = useState({ open: false, data: null });
  const [updateModal, setUpdateModal] = useState({ open: false, data: null });
  const [scheduleModal, setScheduleModal] = useState({ open: false, data: null });
  const [offerModal, setOfferModal] = useState({ open: false, data: null });

  // Data Fetching
  const { data: candidates, isLoading: isCandLoading } = useQuery({
    queryKey: ['candidates'],
    queryFn: fetchCandidates,
  });

  const { data: positions, isLoading: isPosLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: fetchPositions,
  });

  // Mutations
  const updateStageMutation = useMutation({
    mutationFn: ({ id, stage }) => updateCandidateStage(id, stage),
    onSuccess: (data) => {
      message.success(`Candidate stage moved to: ${data.recruitmentStage}`);
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
    },
  });

  const scheduleMutation = useMutation({
    mutationFn: ({ id, data }) => scheduleCandidateInterview(id, data),
    onSuccess: () => {
      message.success('Interview scheduled successfully and invitation sent!');
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
      setScheduleModal({ open: false, data: null });
    },
  });

  const updateCVMutation = useMutation({
    mutationFn: ({ id, data }) => updateCandidateCVInfo(id, data),
    onSuccess: () => {
      message.success('Candidate info updated successfully!');
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
      setUpdateModal({ open: false, data: null });
    },
  });

  const handleStageChange = (id, newStage) => {
    updateStageMutation.mutate({ id, stage: newStage });
  };

  const clearAllFilters = () => {
    setSearchText('');
    setStageFilter(null);
    setPositionFilter(null);
    setTypeFilter(null);
    setSearchParams({}); // Clear URL params
  };

  const getActionMenuItems = (record) => {
    const stage = record.recruitmentStage;
    const items = [];

    if (stage === 'APPLIED') {
      items.push({ key: 'SCHEDULE_INTERVIEW', label: 'Schedule Interview' });
      items.push({ key: 'REJECTED', label: 'Reject', danger: true });
    } else if (stage === 'INTERVIEW_SCHEDULED') {
      items.push({ key: 'INTERVIEWED', label: 'Mark as Interviewed' });
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
        if (item.key === 'SCHEDULE_INTERVIEW' || item.key === 'RE_SCHEDULE') {
          setScheduleModal({ open: true, data: record });
        } else if (item.key === 'OFFER') {
          setOfferModal({ open: true, data: record });
        } else {
          handleStageChange(record.id, item.key);
        }
      }
    }));
  };

  // Filter Logic
  const filteredData = candidates?.filter(item => {
    const matchNameOrEmail = item.name.toLowerCase().includes(searchText.toLowerCase()) || 
                             item.email.toLowerCase().includes(searchText.toLowerCase());
    const matchStage = stageFilter ? item.recruitmentStage === stageFilter : true;
    const matchPosition = positionFilter ? item.position_id === positionFilter : true;
    const matchType = typeFilter ? item.type === typeFilter : true;
    return matchNameOrEmail && matchStage && matchPosition && matchType;
  });

  // Lấy tên Job Title nếu có filter theo Position
  const filteredJobTitle = positionFilter ? positions?.find(p => p.id === positionFilter)?.name : null;

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
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color={type === 'INTERNAL' ? 'geekblue' : 'green'}>{type}</Tag>
      )
    },
    {
      title: 'Applied Date',
      dataIndex: 'updatedAt',
      key: 'updatedAt',
      render: (date) => dayjs(date).format('DD/MM/YYYY')
    },
    {
      title: 'AI Score',
      key: 'score',
      render: (_, record) => <ScoreBadge score={record.analysis?.overallScore} />
    },
    {
      title: 'Reason for Match',
      key: 'reason',
      width: 250,
      render: (_, record) => (
        <Paragraph ellipsis={{ rows: 2, tooltip: record.analysis?.feedback }} style={{ margin: 0, fontSize: 13, color: '#595959' }}>
          {record.analysis?.feedback || '-'}
        </Paragraph>
      )
    },
    {
      title: 'Stage',
      dataIndex: 'recruitmentStage',
      key: 'recruitmentStage',
      render: (stage) => <StageTag stage={stage} />
    },
    {
      title: 'Interview Schedule',
      dataIndex: 'interviewDate',
      key: 'interviewDate',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY HH:mm') : <Text type="secondary">-</Text>
    },
    {
      title: 'Actions',
      key: 'action',
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="View CV">
            <Button type="text" icon={<EyeOutlined />} onClick={() => window.open(record.driveFileUrl, '_blank')} />
          </Tooltip>
          <Tooltip title="AI Analysis">
            <Button type="text" icon={<FileTextOutlined />} onClick={() => setAnalysisModal({ open: true, data: record.analysis })} />
          </Tooltip>
          
          {record.type === 'INTERNAL' && (
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

  if (isCandLoading || isPosLoading) {
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
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          allowClear
        />
        <Select
          allowClear
          placeholder="Filter by Position"
          style={{ width: 250 }}
          value={positionFilter}
          onChange={setPositionFilter}
          options={positions?.map(p => ({ value: p.id, label: p.name })) || []}
        />
        <Select
          allowClear
          placeholder="Filter by Source Type"
          style={{ width: 150 }}
          value={typeFilter}
          onChange={setTypeFilter}
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
          onChange={setStageFilter}
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
        dataSource={filteredData} 
        loading={isCandLoading || updateStageMutation.isPending || scheduleMutation.isPending || updateCVMutation.isPending} 
      />

      <Modal
        title="AI Analysis Details"
        open={analysisModal.open}
        onCancel={() => setAnalysisModal({ open: false, data: null })}
        footer={[<Button key="close" onClick={() => setAnalysisModal({ open: false, data: null })}>Close</Button>]}
      >
        {analysisModal.data ? (
          <div>
            <p><strong>Technical Score:</strong> {analysisModal.data.technicalScore}</p>
            <p><strong>Experience Score:</strong> {analysisModal.data.experienceScore}</p>
            <p><strong>Skill Match:</strong> {analysisModal.data.skillMatch}</p>
            <p><strong>Skill Miss:</strong> {analysisModal.data.skillMiss}</p>
            <p><strong>Reason for match:</strong> {analysisModal.data.feedback}</p>
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
        onSave={(data) => updateCVMutation.mutate({ id: updateModal.data.id, data })}
      />

      <ScheduleInterviewModal
        open={scheduleModal.open}
        onCancel={() => setScheduleModal({ open: false, data: null })}
        candidateData={scheduleModal.data}
        onSave={(data) => scheduleMutation.mutate({ id: scheduleModal.data.id, data })}
      />

      <SendOfferModal
        open={offerModal.open}
        onCancel={() => setOfferModal({ open: false, data: null })}
        candidateData={offerModal.data}
        onSave={(data) => {
          message.success('Offer sent successfully!');
          handleStageChange(offerModal.data.id, 'OFFER');
          setOfferModal({ open: false, data: null });
        }}
      />
    </div>
  );
};

export default CandidatesPage;
