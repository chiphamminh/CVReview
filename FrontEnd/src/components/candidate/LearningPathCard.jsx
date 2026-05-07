import { Card, Typography, Space, Progress, List, Tag } from 'antd';
import { BookOutlined, AimOutlined, WarningOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

const LearningPathCard = ({ score, learningPathText, missingSkills }) => {
  return (
    <Card
      style={{ 
        marginTop: 16, 
        borderColor: '#ffa39e', 
        backgroundColor: '#fff1f0',
        borderRadius: 8 
      }}
      bodyStyle={{ padding: 20 }}
    >
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Progress type="circle" percent={score} size={60} status="exception" format={(p) => `${p} pt`} />
          <div>
            <Title level={5} style={{ margin: 0, color: '#cf1322' }}>
              <WarningOutlined style={{ marginRight: 8 }} />
              Application Not Matched
            </Title>
            <Text type="secondary">
              Your profile score is below the 70 points threshold required for this position.
            </Text>
          </div>
        </div>

        {missingSkills && missingSkills.length > 0 && (
          <div>
            <Text strong><AimOutlined style={{ marginRight: 8 }} />Missing Key Skills:</Text>
            <div style={{ marginTop: 8 }}>
              {missingSkills.map((skill, idx) => (
                <Tag color="volcano" key={idx}>{skill}</Tag>
              ))}
            </div>
          </div>
        )}

        <div>
          <Text strong><BookOutlined style={{ marginRight: 8 }} />Suggested Learning Path:</Text>
          <Paragraph style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>
            {learningPathText || 'We recommend reviewing the core requirements of this role and gaining more hands-on experience before reapplying.'}
          </Paragraph>
        </div>
      </Space>
    </Card>
  );
};

export default LearningPathCard;
