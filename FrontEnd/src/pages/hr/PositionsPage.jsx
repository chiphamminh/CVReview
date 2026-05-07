import React, { useState } from 'react';
import { Button, Space, Switch, InputNumber, Typography, Tooltip, message, Tag, Input, Select, Row, Col } from 'antd';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { PlusOutlined, FileTextOutlined, RobotOutlined, EditOutlined, DeleteOutlined, SearchOutlined, UploadOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { useNavigate } from 'react-router-dom';

import AppTable from '@/components/tables/AppTable';
import LoadingSkeleton from '@/components/common/LoadingSkeleton';
import DeleteWarningPopup from '@/components/modals/DeleteWarningPopup';
import PositionFormModal from '@/components/modals/PositionFormModal';
import UploadCVModal from '@/components/modals/UploadCVModal';
import { fetchPositions, updatePositionScore, togglePositionActive } from '@/api/mockData';

const { Title, Text } = Typography;
const { Option } = Select;

const PositionsPage = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const [deleteVisible, setDeleteVisible] = useState(false);
  const [formVisible, setFormVisible] = useState(false);
  const [uploadCvVisible, setUploadCvVisible] = useState(false);
  const [selectedPos, setSelectedPos] = useState(null);

  // Filters
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState(null); // 'all', 'active', 'closed'

  const { data: positions, isLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: fetchPositions,
  });

  const updateScoreMutation = useMutation({
    mutationFn: ({ id, score }) => updatePositionScore(id, score),
    onSuccess: () => {
      message.success('Minimum Score updated successfully!');
      queryClient.invalidateQueries({ queryKey: ['positions'] });
    },
  });

  const toggleActiveMutation = useMutation({
    mutationFn: ({ id, isActive }) => togglePositionActive(id, isActive),
    onSuccess: () => {
      message.success('Position status updated!');
      queryClient.invalidateQueries({ queryKey: ['positions'] });
    },
  });

  const handleScoreChange = (value, record) => {
    if (value && value !== record.minFitScore) {
      updateScoreMutation.mutate({ id: record.id, score: value });
    }
  };

  const handleToggleActive = (checked, record) => {
    toggleActiveMutation.mutate({ id: record.id, isActive: checked });
  };

  const handleCreateOrEdit = (values) => {
    // API call should be here
    message.success(selectedPos ? 'Position updated!' : 'Position created!');
    setFormVisible(false);
  };

  const navigateToCandidates = (positionId, type) => {
    navigate(`/hr/candidates?positionId=${positionId}&type=${type}`);
  };

  // Áp dụng Filter
  const filteredData = positions?.filter(item => {
    const matchName = item.name.toLowerCase().includes(searchText.toLowerCase());
    let matchStatus = true;
    if (statusFilter === 'active') matchStatus = item.isActive === true;
    if (statusFilter === 'closed') matchStatus = item.isActive === false;
    return matchName && matchStatus;
  });

  const columns = [
    {
      title: 'Job Title',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.level} • {record.language}
          </Text>
        </div>
      )
    },
    {
      title: 'Status',
      dataIndex: 'isActive',
      key: 'isActive',
      render: (isActive, record) => (
        <Switch 
          checked={isActive} 
          checkedChildren="Active" 
          unCheckedChildren="Closed" 
          onChange={(checked) => handleToggleActive(checked, record)}
          loading={toggleActiveMutation.isPending && toggleActiveMutation.variables?.id === record.id}
        />
      ),
    },
    {
      title: 'Min Fit Score',
      dataIndex: 'minFitScore',
      key: 'minFitScore',
      render: (score, record) => (
        <Space>
          <InputNumber
            min={0}
            max={100}
            defaultValue={score}
            onBlur={(e) => handleScoreChange(parseInt(e.target.value), record)}
            onPressEnter={(e) => handleScoreChange(parseInt(e.target.value), record)}
            style={{ width: 60 }}
          />
        </Space>
      ),
    },
    {
      title: 'Candidates',
      key: 'candidates',
      render: (_, record) => (
        <Space direction="vertical" size="small">
          <Tag 
            color="geekblue" 
            style={{ cursor: 'pointer', margin: 0, width: '100%' }}
            onClick={() => navigateToCandidates(record.id, 'INTERNAL')}
          >
            Internal: {record.internalCount}
          </Tag>
          <Tag 
            color="green" 
            style={{ cursor: 'pointer', margin: 0, width: '100%' }}
            onClick={() => navigateToCandidates(record.id, 'EXTERNAL')}
          >
            External: {record.externalCount}
          </Tag>
        </Space>
      ),
    },
    {
      title: 'Opened At',
      dataIndex: 'openedAt',
      key: 'openedAt',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY') : '-',
    },
    {
      title: 'Closed At',
      dataIndex: 'closedAt',
      key: 'closedAt',
      render: (date) => date ? dayjs(date).format('DD/MM/YYYY') : <Text type="secondary">-</Text>,
    },
    {
      title: 'Actions',
      key: 'action',
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Upload CVs for this Position">
            <Button type="text" icon={<UploadOutlined />} style={{ color: '#fa8c16' }} onClick={() => { setSelectedPos(record); setUploadCvVisible(true); }} />
          </Tooltip>
          <Tooltip title="View JD">
            <Button type="text" icon={<FileTextOutlined />} onClick={() => console.log('View JD')} />
          </Tooltip>
          <Tooltip title="Chat with AI for this Position">
            <Button type="text" icon={<RobotOutlined />} style={{ color: '#52c41a' }} onClick={() => navigate(`/hr/chatbot/${record.id}`)} />
          </Tooltip>
          <Tooltip title="Edit Position">
            <Button type="text" icon={<EditOutlined />} onClick={() => { setSelectedPos(record); setFormVisible(true); }} />
          </Tooltip>
          <Tooltip title="Delete">
            <Button type="text" danger icon={<DeleteOutlined />} onClick={() => { setSelectedPos(record); setDeleteVisible(true); }} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  if (isLoading) {
    return <LoadingSkeleton rows={10} />;
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>Positions Management</Title>
        <Space>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => { setSelectedPos(null); setFormVisible(true); }}>
            Create Position
          </Button>
        </Space>
      </div>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Input 
            placeholder="Search by job title..." 
            prefix={<SearchOutlined />} 
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
          />
        </Col>
        <Col span={6}>
          <Select 
            placeholder="Filter by Status" 
            style={{ width: '100%' }} 
            allowClear
            value={statusFilter}
            onChange={setStatusFilter}
          >
            <Option value="active">Active</Option>
            <Option value="closed">Closed</Option>
          </Select>
        </Col>
      </Row>

      <AppTable 
        columns={columns} 
        dataSource={filteredData} 
        loading={isLoading || updateScoreMutation.isPending || toggleActiveMutation.isPending} 
      />

      <PositionFormModal 
        open={formVisible}
        onCancel={() => setFormVisible(false)}
        initialData={selectedPos}
        onSave={handleCreateOrEdit}
      />

      <UploadCVModal
        open={uploadCvVisible}
        onCancel={() => setUploadCvVisible(false)}
        positionName={selectedPos?.name}
      />

      <DeleteWarningPopup
        open={deleteVisible}
        onCancel={() => setDeleteVisible(false)}
        onConfirm={() => {
          message.success(`Position deleted: ${selectedPos?.name}`);
          setDeleteVisible(false);
        }}
        title="Delete Position"
        content={`Are you sure you want to delete position "${selectedPos?.name}"? All applications related to this position will be lost.`}
        confirmText="Delete"
        cancelText="Cancel"
      />
    </div>
  );
};

export default PositionsPage;
