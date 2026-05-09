import React, { useState, useEffect, useRef } from 'react';
import { Modal, Upload, Button, Progress, Typography, Space, Card, Row, Col, App } from 'antd';
import {
  InboxOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  FileOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import useAuthStore from '@/store/authStore';
import { uploadApi } from '@/api/upload.api';

const { Dragger } = Upload;
const { Text, Title } = Typography;

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

const UploadCVModal = ({ open, onCancel, positionId, positionName }) => {
  const { message } = App.useApp();
  const [fileList, setFileList] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);
  const [successCount, setSuccessCount] = useState(0);
  const [failCount, setFailCount] = useState(0);
  const sseAbortRef = useRef(null);

  const stopSSE = () => {
    if (sseAbortRef.current) {
      sseAbortRef.current.abort();
      sseAbortRef.current = null;
    }
  };

  useEffect(() => {
    if (!open) {
      setFileList([]);
      setIsUploading(false);
      setProgress(0);
      setProcessedCount(0);
      setSuccessCount(0);
      setFailCount(0);
      stopSSE();
    }
  }, [open]);

  useEffect(() => () => stopSSE(), []);

  const connectSSE = (batchId, total) => {
    const controller = new AbortController();
    sseAbortRef.current = controller;
    const token = useAuthStore.getState().token;

    fetchEventSource(`${BASE_URL}/tracking/${batchId}/stream`, {
      method: 'GET',
      headers: {
        Authorization: token ? `Bearer ${token}` : '',
        Accept: 'text/event-stream',
      },
      signal: controller.signal,
      openWhenHidden: true,

      onmessage: (event) => {
        if (event.event === 'batch-completed') {
          setProgress(100);
          setIsUploading(false);
          stopSSE();
          return;
        }

        try {
          const status = JSON.parse(event.data);
          const processed = status.processed ?? 0;
          setProcessedCount(processed);
          setSuccessCount(status.success ?? 0);
          setFailCount(status.failed ?? 0);
          setProgress(total > 0 ? Math.round((processed / total) * 100) : 0);

          if (status.status === 'COMPLETED') {
            setProgress(100);
            setIsUploading(false);
            stopSSE();
          }
        } catch {
          // bỏ qua parse error
        }
      },

      onerror: (err) => {
        if (err?.name === 'AbortError') return;
        setIsUploading(false);
        throw err; // dừng retry
      },
    });
  };

  const handleUpload = async () => {
    if (fileList.length === 0 || !positionId) return;
    setIsUploading(true);

    try {
      const res = await uploadApi.hrUploadCVs(positionId, fileList);
      const batchId = res.data?.batchId;
      if (!batchId) throw new Error('No batchId returned from server');
      connectSSE(batchId, fileList.length);
    } catch (err) {
      message.error(err.response?.data?.message || 'Upload failed. Please try again.');
      setIsUploading(false);
    }
  };

  const uploadProps = {
    onRemove: (file) => setFileList((prev) => prev.filter((f) => f !== file)),
    beforeUpload: (file) => {
      setFileList((prev) => [...prev, file]);
      return false;
    },
    fileList,
    multiple: true,
    accept: '.pdf,.doc,.docx',
  };

  const totalCV = fileList.length;
  const pendingCV = totalCV - processedCount;
  const isDone = progress === 100;

  return (
    <Modal
      title={`Upload CVs for ${positionName || 'Position'}`}
      open={open}
      onCancel={() => { if (!isUploading) onCancel(); }}
      maskClosable={!isUploading}
      closable={!isUploading}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={isUploading}>
          {isDone ? 'Close' : 'Cancel'}
        </Button>,
        <Button
          key="submit"
          type="primary"
          onClick={handleUpload}
          disabled={fileList.length === 0 || isUploading || isDone}
          loading={isUploading}
        >
          Start Processing
        </Button>,
      ]}
      width={600}
    >
      {!isUploading && !isDone ? (
        <Dragger {...uploadProps} style={{ marginTop: 16 }}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">Click or drag multiple CVs to upload</p>
          <p className="ant-upload-hint">
            Supports PDF, DOC, DOCX. The system will process each CV automatically.
          </p>
        </Dragger>
      ) : (
        <Card
          title={
            <Space>
              {isUploading ? (
                <SyncOutlined spin style={{ color: '#1677ff' }} />
              ) : (
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              )}
              <span>{isUploading ? 'Processing CVs...' : 'Processing Complete'}</span>
            </Space>
          }
          style={{ marginTop: 16, borderColor: '#1677ff' }}
        >
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <Text strong>Overall Progress</Text>
              <Text strong>{progress}%</Text>
            </div>
            <Progress
              percent={progress}
              showInfo={false}
              status={isDone ? 'success' : 'active'}
              strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
            />
          </div>

          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card size="small" style={{ background: '#f5f5f5' }}>
                <Space>
                  <FileOutlined style={{ fontSize: 20, color: '#1890ff' }} />
                  <div>
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Total CV</Text>
                    <Title level={4} style={{ margin: 0 }}>{totalCV}</Title>
                  </div>
                </Space>
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" style={{ background: '#fffbe6' }}>
                <Space>
                  <ClockCircleOutlined style={{ fontSize: 20, color: '#faad14' }} />
                  <div>
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Pending</Text>
                    <Title level={4} style={{ margin: 0 }}>{pendingCV}</Title>
                  </div>
                </Space>
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" style={{ background: '#f6ffed' }}>
                <Space>
                  <CheckCircleOutlined style={{ fontSize: 20, color: '#52c41a' }} />
                  <div>
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Success</Text>
                    <Title level={4} style={{ margin: 0 }}>{successCount}</Title>
                  </div>
                </Space>
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" style={{ background: '#fff1f0' }}>
                <Space>
                  <CloseCircleOutlined style={{ fontSize: 20, color: '#f5222d' }} />
                  <div>
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Failed</Text>
                    <Title level={4} style={{ margin: 0 }}>{failCount}</Title>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </Card>
      )}
    </Modal>
  );
};

export default UploadCVModal;
