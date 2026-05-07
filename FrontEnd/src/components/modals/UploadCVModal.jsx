import React, { useState, useEffect } from 'react';
import { Modal, Upload, Button, Progress, Typography, Space, Card, Row, Col } from 'antd';
import { InboxOutlined, CheckCircleOutlined, CloseCircleOutlined, SyncOutlined, FileOutlined, ClockCircleOutlined } from '@ant-design/icons';

const { Dragger } = Upload;
const { Text, Title } = Typography;

const UploadCVModal = ({ open, onCancel, positionName }) => {
  const [fileList, setFileList] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  
  // Progress state simulating SSE
  const [progress, setProgress] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);
  const [successCount, setSuccessCount] = useState(0);
  const [failCount, setFailCount] = useState(0);

  // Timer reference
  const [timerId, setTimerId] = useState(null);

  useEffect(() => {
    if (!open) {
      setFileList([]);
      setIsUploading(false);
      setProgress(0);
      setProcessedCount(0);
      setSuccessCount(0);
      setFailCount(0);
      if (timerId) clearInterval(timerId);
    }
  }, [open]);

  const handleUpload = () => {
    if (fileList.length === 0) return;
    setIsUploading(true);

    const totalFiles = fileList.length;
    let currentProcessed = 0;
    let currentSuccess = 0;
    let currentFail = 0;

    const interval = setInterval(() => {
      currentProcessed += 1;
      
      // Randomly fail ~10% of files
      if (Math.random() > 0.9) {
        currentFail += 1;
      } else {
        currentSuccess += 1;
      }

      const currentProgress = Math.floor((currentProcessed / totalFiles) * 100);
      
      setProcessedCount(currentProcessed);
      setSuccessCount(currentSuccess);
      setFailCount(currentFail);
      setProgress(currentProgress);

      if (currentProcessed >= totalFiles) {
        clearInterval(interval);
        setIsUploading(false);
      }
    }, 1500); // 1.5 seconds per file
    
    setTimerId(interval);
  };

  const uploadProps = {
    onRemove: (file) => {
      const index = fileList.indexOf(file);
      const newFileList = fileList.slice();
      newFileList.splice(index, 1);
      setFileList(newFileList);
    },
    beforeUpload: (file) => {
      setFileList(prev => [...prev, file]);
      return false;
    },
    fileList,
    multiple: true,
  };

  const totalCV = fileList.length;
  const pendingCV = totalCV - processedCount;

  return (
    <Modal
      title={`Upload CVs for ${positionName || 'Position'}`}
      open={open}
      onCancel={() => {
        if (!isUploading) onCancel();
      }}
      maskClosable={!isUploading}
      closable={!isUploading}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={isUploading}>
          {progress === 100 ? 'Close' : 'Cancel'}
        </Button>,
        <Button 
          key="submit" 
          type="primary" 
          onClick={handleUpload} 
          disabled={fileList.length === 0 || isUploading || progress === 100}
          loading={isUploading}
        >
          Start Processing
        </Button>
      ]}
      width={600}
    >
      {!isUploading && progress === 0 ? (
        <Dragger {...uploadProps} style={{ marginTop: 16 }}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">Click or drag multiple CVs to this area to upload</p>
          <p className="ant-upload-hint">Support bulk upload. The system will process each CV automatically.</p>
        </Dragger>
      ) : (
        <Card title={
          <Space>
            {isUploading ? <SyncOutlined spin style={{ color: '#1677ff' }} /> : <CheckCircleOutlined style={{ color: '#52c41a' }} />}
            <span>{isUploading ? "Connecting to SSE... Extracting & Analyzing Data" : "Processing Complete"}</span>
          </Space>
        } style={{ marginTop: 16, borderColor: '#1677ff' }}>
          
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <Text strong>Overall Progress</Text>
              <Text strong>{progress}%</Text>
            </div>
            <Progress percent={progress} showInfo={false} status={progress === 100 ? "success" : "active"} strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }} />
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
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Pending CV</Text>
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
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Success CV</Text>
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
                    <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>Fail CV</Text>
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
