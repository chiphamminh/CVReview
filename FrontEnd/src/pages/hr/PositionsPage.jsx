import React, { useState, useEffect, useCallback, useRef } from 'react';

import {
  Button, Space, Switch, InputNumber, Typography, Tooltip,
  Tag, Input, Select, Row, Col, Drawer, App, Modal,
} from 'antd';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  PlusOutlined, FileTextOutlined, RobotOutlined,
  EditOutlined, DeleteOutlined, SearchOutlined, UploadOutlined,
  LinkOutlined, SyncOutlined, CheckCircleOutlined, ExclamationCircleOutlined, ReloadOutlined,
} from '@ant-design/icons';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import ReactMarkdown from 'react-markdown';
import dayjs from 'dayjs';
import { useNavigate } from 'react-router-dom';

import AppTable from '@/components/tables/AppTable';
import LoadingSkeleton from '@/components/common/LoadingSkeleton';
import DeleteWarningPopup from '@/components/modals/DeleteWarningPopup';
import PositionFormModal from '@/components/modals/PositionFormModal';
import UploadCVModal from '@/components/modals/UploadCVModal';
import { positionApi } from '@/api/position.api';
import useAuthStore from '@/store/authStore';

const { Title, Text } = Typography;
const { Option } = Select;

const PAGE_SIZE = 10;

const PositionsPage = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { message, notification } = App.useApp();
  const jdSseControllerRef = useRef(null);
  const jdTimerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (jdSseControllerRef.current) jdSseControllerRef.current.abort();
      if (jdTimerRef.current) clearInterval(jdTimerRef.current);
    };
  }, []);

  const [deleteVisible, setDeleteVisible] = useState(false);
  const [formVisible, setFormVisible] = useState(false);
  const [uploadCvVisible, setUploadCvVisible] = useState(false);
  const [jdDrawer, setJdDrawer] = useState({ visible: false, title: '', text: '', driveUrl: '' });
  const [selectedPos, setSelectedPos] = useState(null);

  // Filter state
  const [searchInput, setSearchInput] = useState('');
  const [keyword, setKeyword] = useState('');
  const [statusFilter, setStatusFilter] = useState(null);
  const [page, setPage] = useState(0);

  // Debounce keyword
  useEffect(() => {
    const timer = setTimeout(() => {
      setKeyword(searchInput);
      setPage(0);
    }, 500);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const isActive = statusFilter === 'active' ? true : statusFilter === 'closed' ? false : undefined;

  const { data: positionsData, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['positions', { keyword, isActive, page }],
    queryFn: () => positionApi.filter({ keyword: keyword || undefined, isActive, page, size: PAGE_SIZE }),
  });

  const positions = positionsData?.data?.content ?? [];
  const totalElements = positionsData?.data?.totalElements ?? 0;

  // ── Mutations ──────────────────────────────────────────────────────────────

  const createMutation = useMutation({
    mutationFn: (fd) => positionApi.create(fd),
    onSuccess: (res) => {
      const { batchId, title } = res.data ?? {};
      setFormVisible(false);
      trackJDProcessing(batchId, title);
    },
    onError: (err) => message.error(err.response?.data?.message || 'Failed to create position'),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, formData }) => positionApi.update(id, formData),
    onSuccess: () => {
      message.success('Position updated!');
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      setFormVisible(false);
    },
    onError: (err) => message.error(err.response?.data?.message || 'Failed to update position'),
  });

  const deleteMutation = useMutation({
    mutationFn: (ids) => positionApi.deleteMany(ids),
    onSuccess: () => {
      message.success('Position deleted!');
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      setDeleteVisible(false);
    },
    onError: (err) =>
      message.error(err.response?.data?.message || 'Cannot delete — position has linked candidates'),
  });

  const updateScoreMutation = useMutation({
    mutationFn: ({ id, score }) => positionApi.updateMinScore(id, score),
    onSuccess: () => {
      message.success('Minimum score updated!');
      queryClient.invalidateQueries({ queryKey: ['positions'] });
    },
    onError: (err) => message.error(err.response?.data?.message || 'Failed to update score'),
  });

  const toggleActiveMutation = useMutation({
    mutationFn: ({ id }) => positionApi.toggleActive(id),
    onSuccess: (_, { wasActive }) => {
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      if (wasActive) {
        message.success('Position closed — hidden from candidates.');
      } else {
        message.success('Position activated — visible to candidates.');
      }
    },
    onError: (err) => message.error(err.response?.data?.message || 'Failed to update status'),
  });

  // ── Handlers ───────────────────────────────────────────────────────────────

  const trackJDProcessing = useCallback(
    (batchId, positionTitle) => {
      if (!batchId) {
        message.success('Position created successfully!');
        queryClient.invalidateQueries({ queryKey: ['positions'] });
        return;
      }

      const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';
      const token = useAuthStore.getState().token;
      const key = `jd-${batchId}`;

      if (jdTimerRef.current) clearInterval(jdTimerRef.current);

      notification.open({
        key,
        message: 'Processing Job Description',
        description: `Parsing and embedding JD for "${positionTitle}"... (0%)`,
        icon: <SyncOutlined spin style={{ color: '#1677ff' }} />,
        duration: 0,
      });

      let simulatedProgress = 0;
      jdTimerRef.current = setInterval(() => {
        if (simulatedProgress < 90) {
          simulatedProgress += Math.max(1, Math.floor((90 - simulatedProgress) * 0.1));
        }
        notification.open({
          key,
          message: 'Processing Job Description',
          description: `Parsing and embedding JD for "${positionTitle}"... (${simulatedProgress}%)`,
          icon: <SyncOutlined spin style={{ color: '#1677ff' }} />,
          duration: 0,
        });
      }, 1000);

      if (jdSseControllerRef.current) jdSseControllerRef.current.abort();
      const controller = new AbortController();
      jdSseControllerRef.current = controller;

      const finish = (success) => {
        clearInterval(jdTimerRef.current);
        jdTimerRef.current = null;
        if (success) {
          notification.open({
            key,
            message: 'JD Ready',
            description: `"${positionTitle}" is processed and ready for candidate matching.`,
            icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
            duration: 5,
          });
          queryClient.invalidateQueries({ queryKey: ['positions'] });
        } else {
          notification.open({
            key,
            message: 'JD Processing Failed',
            description: `Failed to process JD for "${positionTitle}". Please check and retry.`,
            icon: <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
            duration: 6,
          });
        }
        controller.abort();
      };

      fetchEventSource(`${BASE_URL}/tracking/${batchId}/stream`, {
        method: 'GET',
        headers: {
          Authorization: token ? `Bearer ${token}` : '',
          Accept: 'text/event-stream',
        },
        signal: controller.signal,
        openWhenHidden: true,

        onmessage: (event) => {
          if (event.event === 'batch-completed') { finish(true); return; }
          try {
            const status = JSON.parse(event.data);
            if (status.status === 'COMPLETED') finish(true);
          } catch { /* ignore */ }
        },

        onerror: (err) => {
          if (err?.name === 'AbortError') return;
          finish(false);
          throw err;
        },
      });

      queryClient.invalidateQueries({ queryKey: ['positions'] });
    },
    [notification, message, queryClient, jdSseControllerRef, jdTimerRef]
  );

  const handleToggleActive = useCallback(
    (currentlyActive, record) => {
      if (currentlyActive) {
        Modal.confirm({
          title: 'Close this position?',
          content: `"${record.title}" will be hidden from candidates and no new applications will be accepted.`,
          okText: 'Close Position',
          okButtonProps: { danger: true },
          cancelText: 'Cancel',
          onOk: () => toggleActiveMutation.mutate({ id: record.id, wasActive: true }),
        });
      } else {
        toggleActiveMutation.mutate({ id: record.id, wasActive: false });
      }
    },
    [toggleActiveMutation]
  );

  const handleCreateOrEdit = useCallback(
    (values) => {
      const fd = new FormData();
      fd.append('title', values.title);
      fd.append('seniority', values.seniority);
      (values.skills ?? []).forEach((s) => fd.append('skills', s));
      if (!selectedPos && values.file?.[0]?.originFileObj) {
        fd.append('file', values.file[0].originFileObj);
      }
      if (selectedPos) {
        updateMutation.mutate({ id: selectedPos.id, formData: fd });
      } else {
        createMutation.mutate(fd);
      }
    },
    [selectedPos, createMutation, updateMutation]
  );

  const handleViewJD = useCallback(async (record) => {
    try {
      const res = await positionApi.getJDText(record.id);
      setJdDrawer({
        visible: true,
        title: record.title,
        text: res.data?.jdText || 'No JD content available.',
        driveUrl: record.driveFileUrl || '',
      });
    } catch {
      message.error('Failed to load JD content');
    }
  }, [message]);

  const handleScoreBlurOrEnter = useCallback(
    (value, record) => {
      const num = parseFloat(value);
      if (!isNaN(num) && num !== record.minimumFitScore) {
        updateScoreMutation.mutate({ id: record.id, score: num });
      }
    },
    [updateScoreMutation]
  );

  const navigateToCandidates = useCallback(
    (positionId, sourceType) => {
      navigate(`/hr/candidates?positionId=${positionId}&sourceType=${sourceType}`);
    },
    [navigate]
  );

  // ── Table Columns ──────────────────────────────────────────────────────────

  const columns = [
    {
      title: 'Job Title',
      dataIndex: 'title',
      key: 'title',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.seniority}
            {record.skills?.length ? ` • ${record.skills.slice(0, 3).join(', ')}` : ''}
          </Text>
        </div>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'isActive',
      key: 'isActive',
      width: 110,
      render: (active, record) => (
        <Switch
          checked={active}
          checkedChildren="Active"
          unCheckedChildren="Closed"
          onChange={() => handleToggleActive(active, record)}
          loading={toggleActiveMutation.isPending && toggleActiveMutation.variables?.id === record.id}
        />
      ),
    },
    {
      title: 'Min Fit Score',
      dataIndex: 'minimumFitScore',
      key: 'minimumFitScore',
      width: 120,
      render: (score, record) => (
        <InputNumber
          key={`${record.id}-${score}`}
          min={0}
          max={100}
          defaultValue={score ?? 70}
          onBlur={(e) => handleScoreBlurOrEnter(e.target.value, record)}
          onPressEnter={(e) => handleScoreBlurOrEnter(e.target.value, record)}
          style={{ width: 70 }}
        />
      ),
    },
    {
      title: 'Candidates',
      key: 'candidates',
      width: 130,
      render: (_, record) => (
        <Space direction="vertical" size={4}>
          <Tag
            color="geekblue"
            style={{ cursor: 'pointer', margin: 0, width: '100%', textAlign: 'center' }}
            onClick={() => navigateToCandidates(record.id, 'INTERNAL')}
          >
            HR Upload: {record.internalCount ?? 0}
          </Tag>
          <Tag
            color="green"
            style={{ cursor: 'pointer', margin: 0, width: '100%', textAlign: 'center' }}
            onClick={() => navigateToCandidates(record.id, 'EXTERNAL')}
          >
            Applied: {record.externalCount ?? 0}
          </Tag>
        </Space>
      ),
    },
    {
      title: 'Opened At',
      dataIndex: 'openedAt',
      key: 'openedAt',
      width: 110,
      render: (date) => (date ? dayjs(date).format('DD/MM/YYYY') : '-'),
    },
    {
      title: 'Closed At',
      dataIndex: 'closedAt',
      key: 'closedAt',
      width: 110,
      render: (date) =>
        date ? dayjs(date).format('DD/MM/YYYY') : <Text type="secondary">-</Text>,
    },
    {
      title: 'Actions',
      key: 'action',
      width: 180,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Upload CVs for this Position">
            <Button
              type="text"
              icon={<UploadOutlined />}
              style={{ color: '#fa8c16' }}
              onClick={() => { setSelectedPos(record); setUploadCvVisible(true); }}
            />
          </Tooltip>
          <Tooltip title="View JD">
            <Button
              type="text"
              icon={<FileTextOutlined />}
              onClick={() => handleViewJD(record)}
            />
          </Tooltip>
          <Tooltip title="Chat with AI for this Position">
            <Button
              type="text"
              icon={<RobotOutlined />}
              style={{ color: '#52c41a' }}
              onClick={() => navigate(`/hr/chatbot/${record.id}`)}
            />
          </Tooltip>
          <Tooltip title="Edit Position">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => { setSelectedPos(record); setFormVisible(true); }}
            />
          </Tooltip>
          <Tooltip title="Delete">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => { setSelectedPos(record); setDeleteVisible(true); }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  if (isLoading && positions.length === 0) {
    return <LoadingSkeleton rows={10} />;
  }

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16, marginRight: 10, marginTop: 10 }}>
        <Title level={4} style={{ margin: '10px 0 0 10px' }}>Positions Management</Title>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => { setSelectedPos(null); setFormVisible(true); }}
        >
          Create Position
        </Button>
      </div>

      {/* Filters */}
      <Row gutter={16} style={{ marginBottom: 16 }} align="middle">
        <Col span={10}>
          <Input
            placeholder="Search by title, seniority, or skills..."
            prefix={<SearchOutlined />}
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            allowClear
          />
        </Col>
        <Col span={6}>
          <Select
            placeholder="Filter by Status"
            style={{ width: '100%' }}
            allowClear
            value={statusFilter}
            onChange={(val) => { setStatusFilter(val); setPage(0); }}
          >
            <Option value="active">Active</Option>
            <Option value="closed">Closed</Option>
          </Select>
        </Col>
        <Col flex="auto" style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button icon={<ReloadOutlined />} onClick={() => refetch()} loading={isFetching}>
            Refresh
          </Button>
        </Col>
      </Row>

      {/* Table */}
      <AppTable
        columns={columns}
        dataSource={positions}
        loading={isLoading}
        rowKey="id"
        pagination={{
          current: page + 1,
          pageSize: PAGE_SIZE,
          total: totalElements,
          onChange: (p) => setPage(p - 1),
          showSizeChanger: false,
          showTotal: (total) => `Total ${total} positions`,
        }}
      />

      {/* Create / Edit Modal */}
      <PositionFormModal
        open={formVisible}
        onCancel={() => setFormVisible(false)}
        initialData={selectedPos}
        onSave={handleCreateOrEdit}
        loading={createMutation.isPending || updateMutation.isPending}
      />

      {/* Upload CV Modal */}
      <UploadCVModal
        open={uploadCvVisible}
        onCancel={() => setUploadCvVisible(false)}
        positionId={selectedPos?.id}
        positionName={selectedPos?.title}
      />

      {/* Delete Confirmation */}
      <DeleteWarningPopup
        open={deleteVisible}
        onCancel={() => setDeleteVisible(false)}
        onConfirm={() => deleteMutation.mutate([selectedPos?.id])}
        title="Delete Position"
        content={`Are you sure you want to delete "${selectedPos?.title}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        loading={deleteMutation.isPending}
      />

      {/* JD Viewer Drawer */}
      <Drawer
        title={
          <Space>
            <span>Job Description — {jdDrawer.title}</span>
            {jdDrawer.driveUrl && (
              <Tooltip title="Open original file in Google Drive">
                <Button
                  type="link"
                  icon={<LinkOutlined />}
                  size="small"
                  href={jdDrawer.driveUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ padding: 0 }}
                >
                  Open in Drive
                </Button>
              </Tooltip>
            )}
          </Space>
        }
        open={jdDrawer.visible}
        onClose={() => setJdDrawer((prev) => ({ ...prev, visible: false }))}
        width={600}
      >
        <div className="jd-markdown-body">
          <ReactMarkdown>{jdDrawer.text}</ReactMarkdown>
        </div>
      </Drawer>
    </div>
  );
};

export default PositionsPage;
