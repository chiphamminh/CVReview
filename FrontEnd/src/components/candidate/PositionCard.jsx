import { useState } from 'react';
import { Card, Button, Typography, Space, Tag, Badge, Modal, Divider } from 'antd';
import { CalendarOutlined, FileTextOutlined, ArrowRightOutlined, CheckCircleOutlined, BookOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { useNavigate } from 'react-router-dom';
import useAuthStore from '@/store/authStore';
import LearningPathCard from '@/components/candidate/LearningPathCard';

const { Title, Text } = Typography;

const getStatusColor = (status) => {
  switch (status) {
    case 'EXCELLENT_MATCH': return 'success';
    case 'GOOD_MATCH': return 'processing';
    case 'POTENTIAL': return 'warning';
    case 'POOR_FIT': return 'error';
    default: return 'default';
  }
};

const formatStatusText = (status) => {
  if (!status) return 'UNKNOWN';
  return status.replace('_', ' ');
};

const PositionCard = ({ position }) => {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const { cvAnalysis } = position;
  
  const [isLearningPathModalOpen, setIsLearningPathModalOpen] = useState(false);

  const handleApply = () => {
    if (!user) {
      navigate('/login');
      return;
    }
    // Navigate to chatbot to finalize application
    navigate(`/candidate/chatbot?positionId=${position.id}&action=apply`);
  };

  const renderAction = () => {
    if (!user) {
      return (
        <Button type="primary" icon={<ArrowRightOutlined />} onClick={() => navigate('/login')} block size="large">
          Login to Apply
        </Button>
      );
    }

    if (!cvAnalysis) {
      return (
        <Button type="default" onClick={() => navigate('/candidate/cv')} block size="large">
          Evaluate Fit (Upload CV)
        </Button>
      );
    }

    if (cvAnalysis.isApplied) {
      return (
        <Button type="primary" icon={<CheckCircleOutlined />} disabled block size="large" style={{ backgroundColor: '#52c41a', borderColor: '#52c41a' }}>
          APPLIED
        </Button>
      );
    }

    if (cvAnalysis.score >= 70) {
      return (
        <Button type="primary" icon={<ArrowRightOutlined />} onClick={handleApply} block size="large">
          Apply Now
        </Button>
      );
    }

    // Score < 70
    return (
      <>
        <Button danger icon={<BookOutlined />} block size="large" onClick={() => setIsLearningPathModalOpen(true)}>
          Learning Path
        </Button>
        <Modal
          title="Suggested Learning Path"
          open={isLearningPathModalOpen}
          onCancel={() => setIsLearningPathModalOpen(false)}
          footer={[
            <Button key="close" onClick={() => setIsLearningPathModalOpen(false)}>
              Close
            </Button>
          ]}
          width={600}
        >
          <LearningPathCard 
            score={cvAnalysis.score} 
            missingSkills={cvAnalysis.missingSkills} 
            learningPathText={cvAnalysis.learningPath} 
          />
        </Modal>
      </>
    );
  };

  return (
    <Card 
      hoverable 
      style={{ borderRadius: 8, border: '1px solid #f0f0f0', height: '100%', display: 'flex', flexDirection: 'column' }}
      styles={{ body: { padding: '24px', display: 'flex', flexDirection: 'column', flex: 1 } }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <Title level={4} style={{ margin: 0, color: '#1677ff', lineHeight: 1.2 }}>{position.name}</Title>
        {position.isHot && <Tag color="volcano" style={{ margin: 0 }}>HOT</Tag>}
      </div>
      
      <Space size={[0, 8]} wrap style={{ marginBottom: 16 }}>
        <Tag color="blue">{position.level}</Tag>
        <Tag color="cyan">{position.language}</Tag>
      </Space>

      <Text type="secondary" style={{ marginBottom: 16, display: 'block' }}>
        <CalendarOutlined style={{ marginRight: 8 }} />
        Posted: {dayjs(position.openDate).format('DD MMM YYYY')}
      </Text>

      {cvAnalysis && (
        <div style={{ padding: '12px', background: '#fafafa', borderRadius: 6, marginBottom: 16 }}>
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary" style={{ fontSize: 13 }}>Match Status</Text>
              <Badge status={getStatusColor(cvAnalysis.overallStatus)} text={<Text strong>{formatStatusText(cvAnalysis.overallStatus)}</Text>} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary" style={{ fontSize: 13 }}>Score</Text>
              <Text strong style={{ color: cvAnalysis.score >= 70 ? '#52c41a' : '#cf1322' }}>{cvAnalysis.score} / 100</Text>
            </div>
          </Space>
        </div>
      )}

      {/* Spacer to push buttons to the bottom */}
      <div style={{ flex: 1 }}></div>

      <Divider style={{ margin: '16px 0' }} />

      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {renderAction()}
        <Button icon={<FileTextOutlined />} onClick={() => window.open(position.jdUrl, '_blank')} block>
          View Job Description
        </Button>
      </Space>
    </Card>
  );
};

export default PositionCard;
