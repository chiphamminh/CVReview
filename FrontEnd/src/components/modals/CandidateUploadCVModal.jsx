import { useState, useEffect, useRef } from 'react';
import { Modal, Upload, Button, Typography, App, Progress, Card, Row, Col, Space } from 'antd';
import {
  InboxOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import useAuthStore from '@/store/authStore';
import { uploadApi } from '@/api/upload.api';

const { Dragger } = Upload;
const { Text } = Typography;

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

const CandidateUploadCVModal = ({ open, onCancel, onSuccess, isReupload }) => {
  const { message } = App.useApp();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isDone, setIsDone] = useState(false);
  const [succeeded, setSucceeded] = useState(false);

  const sseAbortRef = useRef(null);
  const timerRef = useRef(null);

  const stopSSE = () => {
    sseAbortRef.current?.abort();
    sseAbortRef.current = null;
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const startTimer = () => {
    timerRef.current = setInterval(() => {
      setProgress((prev) => (prev < 90 ? prev + Math.max(1, Math.floor((90 - prev) * 0.05)) : prev));
    }, 1000);
  };

  useEffect(() => {
    if (!open) {
      setFile(null);
      setUploading(false);
      setProgress(0);
      setIsDone(false);
      setSucceeded(false);
      stopSSE();
      stopTimer();
    }
    return () => { stopSSE(); stopTimer(); };
  }, [open]);

  const connectSSE = (batchId) => {
    const controller = new AbortController();
    sseAbortRef.current = controller;
    const token = useAuthStore.getState().token;
    startTimer();

    fetchEventSource(`${BASE_URL}/tracking/${batchId}/stream`, {
      method: 'GET',
      headers: { Authorization: token ? `Bearer ${token}` : '', Accept: 'text/event-stream' },
      signal: controller.signal,
      openWhenHidden: true,

      onmessage: (event) => {
        if (event.event === 'batch-completed') {
          setProgress(100);
          setIsDone(true);
          setSucceeded(true);
          setUploading(false);
          stopSSE();
          stopTimer();
          return;
        }
        try {
          const status = JSON.parse(event.data);
          const processed = status.processed ?? 0;
          const actualProgress = processed > 0 ? 100 : 50;
          setProgress((prev) => Math.max(prev, actualProgress));
          if (status.status === 'COMPLETED') {
            setProgress(100);
            setIsDone(true);
            setSucceeded(status.failed === 0);
            setUploading(false);
            stopSSE();
            stopTimer();
          }
          if (status.status === 'FAILED') {
            setIsDone(true);
            setSucceeded(false);
            setUploading(false);
            stopSSE();
            stopTimer();
          }
        } catch { /* ignore parse errors */ }
      },

      onerror: (err) => {
        if (err?.name === 'AbortError') return;
        setUploading(false);
        stopTimer();
        throw err;
      },
    });
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(10);
    try {
      const res = await uploadApi.candidateUploadCV(file);
      const batchId = res.data?.batchId;
      if (!batchId) throw new Error('No batchId returned');
      connectSSE(batchId);
    } catch (err) {
      message.error(err.response?.data?.message || 'Upload failed. Please try again.');
      setUploading(false);
      setProgress(0);
      stopTimer();
    }
  };

  const handleClose = () => {
    if (isDone && succeeded) onSuccess?.();
    onCancel();
  };

  const uploadProps = {
    onRemove: () => setFile(null),
    beforeUpload: (f) => { setFile(f); return false; },
    fileList: file ? [file] : [],
    multiple: false,
    accept: '.pdf,.doc,.docx',
  };

  const title = isReupload ? 'Re-upload Master CV' : 'Upload Master CV';

  return (
    <Modal
      title={title}
      open={open}
      onCancel={() => { if (!uploading) handleClose(); }}
      maskClosable={!uploading}
      closable={!uploading}
      footer={[
        <Button key="cancel" onClick={handleClose} disabled={uploading}>
          {isDone ? 'Close' : 'Cancel'}
        </Button>,
        !isDone && (
          <Button key="upload" type="primary" onClick={handleUpload} disabled={!file || uploading} loading={uploading}>
            Upload
          </Button>
        ),
      ].filter(Boolean)}
      width={520}
    >
      {!uploading && !isDone ? (
        <>
          {isReupload && (
            <div style={{ marginBottom: 16, padding: '10px 14px', background: '#fff7e6', border: '1px solid #ffd591', borderRadius: 6 }}>
              <Text type="warning">Re-uploading will invalidate all your current active applications.</Text>
            </div>
          )}
          <Dragger {...uploadProps}>
            <p className="ant-upload-drag-icon"><InboxOutlined /></p>
            <p className="ant-upload-text">Click or drag your CV here</p>
            <p className="ant-upload-hint">Supports PDF, DOC, DOCX. Single file only.</p>
          </Dragger>
        </>
      ) : (
        <Card
          title={
            <Space>
              {uploading
                ? <SyncOutlined spin style={{ color: '#1677ff' }} />
                : succeeded
                  ? <CheckCircleOutlined style={{ color: '#52c41a' }} />
                  : <CloseCircleOutlined style={{ color: '#f5222d' }} />}
              <span>
                {uploading ? 'Processing CV...' : succeeded ? 'Processing Complete' : 'Processing Failed'}
              </span>
            </Space>
          }
          style={{ marginTop: 8, borderColor: '#1677ff' }}
        >
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <Text strong>Progress</Text>
              <Text strong>{progress}%</Text>
            </div>
            <Progress
              percent={progress}
              showInfo={false}
              status={isDone ? (succeeded ? 'success' : 'exception') : 'active'}
              strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
            />
          </div>

          <Row gutter={[12, 12]}>
            <Col span={24}>
              <Card size="small" style={{ background: isDone ? (succeeded ? '#f6ffed' : '#fff1f0') : '#fffbe6' }}>
                <Space>
                  {isDone
                    ? succeeded
                      ? <CheckCircleOutlined style={{ fontSize: 18, color: '#52c41a' }} />
                      : <CloseCircleOutlined style={{ fontSize: 18, color: '#f5222d' }} />
                    : <SyncOutlined spin style={{ fontSize: 18, color: '#faad14' }} />}
                  <div>
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Status</Text>
                    <Text strong>{isDone ? (succeeded ? 'Done' : 'Failed') : 'In Progress'}</Text>
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

export default CandidateUploadCVModal;
