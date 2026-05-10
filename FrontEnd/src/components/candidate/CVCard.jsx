import { Card, Descriptions, Button, Space, Typography, Tag, Spin } from 'antd';
import { DeleteOutlined, EyeOutlined, EditOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';

const { Title, Text } = Typography;

const SUCCESSFUL_STATUSES = ['EXTRACTED', 'EMBEDDED'];

const STAGE_CONFIG = {
  APPLIED: { color: 'blue', label: 'Applied' },
  INTERVIEW_SCHEDULED: { color: 'orange', label: 'Interview Scheduled' },
  INTERVIEWED: { color: 'purple', label: 'Interviewed' },
  OFFER: { color: 'cyan', label: 'Offer Sent' },
  ACCEPTED: { color: 'green', label: 'Accepted' },
  REJECTED: { color: 'red', label: 'Rejected' },
};

const CVCard = ({ cvData, loading, applications, onDeleteClick, onEditClick }) => {
  if (loading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (!cvData) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Title level={4}>No Master CV Found</Title>
          <Text type="secondary">You haven't uploaded a Master CV yet. Use the button below to upload.</Text>
        </div>
      </Card>
    );
  }

  return (
    <Card
      title={<Title level={4} style={{ margin: 0 }}>My Master CV</Title>}
      extra={
        <Space>
          {cvData.driveFileUrl && (
            <Button icon={<EyeOutlined />} onClick={() => window.open(cvData.driveFileUrl, '_blank')}>
              View CV
            </Button>
          )}
          <Button icon={<EditOutlined />} onClick={onEditClick}>
            Edit Info
          </Button>
          <Button danger icon={<DeleteOutlined />} onClick={onDeleteClick}>
            Delete
          </Button>
        </Space>
      }
    >
      <Descriptions column={{ xs: 1, sm: 2, md: 3 }}>
        <Descriptions.Item label="Name">{cvData.name}</Descriptions.Item>
        <Descriptions.Item label="Email">{cvData.email}</Descriptions.Item>
        <Descriptions.Item label="Upload Date">
          {cvData.updatedAt ? dayjs(cvData.updatedAt).format('DD MMM YYYY, HH:mm') : 'N/A'}
        </Descriptions.Item>
        <Descriptions.Item label="Status">
          <Tag color={SUCCESSFUL_STATUSES.includes(cvData.status) ? 'green' : 'orange'}>
            {cvData.status}
          </Tag>
        </Descriptions.Item>
      </Descriptions>

      <div style={{ marginTop: '24px' }}>
        <Title level={5}>Recent Applications</Title>
        {applications && applications.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {applications.map((app) => {
              const cfg = STAGE_CONFIG[app.recruitmentStage] || { color: 'default', label: app.recruitmentStage };
              return (
                <div
                  key={app.cvId}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: '#fafafa', borderRadius: 6, border: '1px solid #f0f0f0' }}
                >
                  <Text strong>{app.positionTitle || 'Unknown Position'}</Text>
                  <Space size={8}>
                    <Tag color={cfg.color} style={{ margin: 0 }}>{cfg.label}</Tag>
                    {app.recruitmentStage === 'INTERVIEW_SCHEDULED' && app.interviewSchedule && (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {dayjs(app.interviewSchedule).format('DD MMM YYYY, HH:mm')}
                      </Text>
                    )}
                  </Space>
                </div>
              );
            })}
          </div>
        ) : (
          <Text type="secondary">No active applications found.</Text>
        )}
      </div>
    </Card>
  );
};

export default CVCard;
