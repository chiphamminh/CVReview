import { Card, Descriptions, Button, Space, Typography, Tag, Upload } from 'antd';
import { UploadOutlined, DeleteOutlined, EyeOutlined, EditOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';

const { Title, Text } = Typography;

const CVCard = ({ cvData, onUpdateClick, onDeleteClick, onEditClick }) => {
  if (!cvData) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Title level={4}>No Master CV Found</Title>
          <Text type="secondary">You haven't uploaded a Master CV yet.</Text>
          <div style={{ marginTop: '20px' }}>
            <Button type="primary" icon={<UploadOutlined />} onClick={onUpdateClick}>
              Upload Master CV
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      title={<Title level={4} style={{ margin: 0 }}>My Master CV</Title>}
      extra={
        <Space>
          <Button icon={<EyeOutlined />} onClick={() => window.open(cvData.driveFileUrl, '_blank')}>
            View CV
          </Button>
          <Button icon={<EditOutlined />} onClick={onEditClick}>
            Edit Info
          </Button>
          <Button type="primary" icon={<UploadOutlined />} onClick={onUpdateClick}>
            Update CV
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
          <Tag color={cvData.cvStatus === 'PARSED' ? 'green' : 'orange'}>
            {cvData.cvStatus}
          </Tag>
        </Descriptions.Item>
      </Descriptions>
      
      <div style={{ marginTop: '24px' }}>
        <Title level={5}>Recent Applications</Title>
        {cvData.applications && cvData.applications.length > 0 ? (
          <ul>
            {cvData.applications.map((app, idx) => (
              <li key={idx} style={{ marginBottom: 8 }}>
                <Text strong>{app.positionName}</Text> - <Tag>{app.stage}</Tag> 
                <Text type="secondary"> (Applied: {dayjs(app.date).format('DD MMM YYYY')})</Text>
              </li>
            ))}
          </ul>
        ) : (
          <Text type="secondary">No applications found with this CV.</Text>
        )}
      </div>
    </Card>
  );
};

export default CVCard;
