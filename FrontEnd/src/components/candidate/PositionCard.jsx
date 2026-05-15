import { useState } from 'react';
import { Card, Button, Typography, Space, Tag, Badge, Modal, Divider, Spin, Tooltip } from 'antd';
import {
  CalendarOutlined,
  FileTextOutlined,
  ArrowRightOutlined,
  FilePdfOutlined,
  RobotOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import useAuthStore from '@/store/authStore';
import useUiStore from '@/store/uiStore';
import LearningPathCard from '@/components/candidate/LearningPathCard';
import { positionApi } from '@/api/position.api';

const { Title, Text } = Typography;

const STAGE_CONFIG = {
  APPLIED: { color: 'blue', label: 'Applied' },
  INTERVIEW_SCHEDULED: { color: 'orange', label: 'Interview Scheduled' },
  INTERVIEWED: { color: 'purple', label: 'Interviewed' },
  OFFER: { color: 'cyan', label: 'Offer Sent' },
  ACCEPTED: { color: 'green', label: 'Accepted' },
  REJECTED: { color: 'red', label: 'Rejected' },
};

const STAGE_HEX = {
  APPLIED: '#1677ff',
  INTERVIEW_SCHEDULED: '#fa8c16',
  INTERVIEWED: '#531dab',
  OFFER: '#13c2c2',
  ACCEPTED: '#52c41a',
  REJECTED: '#f5222d',
};

const MATCH_STATUS_COLOR = {
  EXCELLENT_MATCH: 'success',
  GOOD_MATCH: 'processing',
  POTENTIAL: 'warning',
  POOR_FIT: 'error',
};

const formatStatus = (s) => s?.replace(/_/g, ' ') ?? '';

const PositionCard = ({ position, applicationData }) => {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const { hasMasterCV } = useAuthStore();
  const { openChatbot } = useUiStore();

  const [isLearningPathModalOpen, setIsLearningPathModalOpen] = useState(false);
  const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
  const [isJDModalOpen, setIsJDModalOpen] = useState(false);
  const [jdText, setJdText] = useState(null);
  const [jdLoading, setJdLoading] = useState(false);

  const handleOpenJD = async () => {
    setIsJDModalOpen(true);
    if (jdText !== null) return;
    setJdLoading(true);
    try {
      const res = await positionApi.getJDText(position.id);
      setJdText(res.data?.jdText || '');
    } catch {
      setJdText('');
    } finally {
      setJdLoading(false);
    }
  };

  const renderAction = () => {
    // Has applied — show stage-based button
    if (applicationData) {
      const stage = applicationData.recruitmentStage;
      const cfg = STAGE_CONFIG[stage] || { color: 'default', label: stage };

      if (stage === 'REJECTED') {
        return (
          <Button block size="large" style={{ borderColor: STAGE_HEX.REJECTED, color: STAGE_HEX.REJECTED, pointerEvents: 'none', cursor: 'default' }}>
            Not Qualified
          </Button>
        );
      }
      if (stage === 'ACCEPTED') {
        return (
          <Button block size="large" style={{ borderColor: STAGE_HEX.ACCEPTED, color: STAGE_HEX.ACCEPTED, pointerEvents: 'none', cursor: 'default' }}>
            Offer Accepted
          </Button>
        );
      }
      return (
        <Button block size="large" style={{ borderColor: STAGE_HEX[stage], color: STAGE_HEX[stage], pointerEvents: 'none', cursor: 'default' }}>
          {cfg.label}
        </Button>
      );
    }

    // Not logged in
    if (!user) {
      return (
        <Button type="primary" icon={<ArrowRightOutlined />} onClick={() => navigate('/login')} block size="large">
          Login to Apply
        </Button>
      );
    }

    // No CV
    if (hasMasterCV === false) {
      return (
        <Button type="default" onClick={() => navigate('/candidate/cv')} block size="large">
          Upload CV to Apply
        </Button>
      );
    }

    // Has CV → open chatbot
    if (hasMasterCV === true) {
      return (
        <Button type="primary" icon={<RobotOutlined />} onClick={openChatbot} block size="large">
          Chat to Evaluate Fit
        </Button>
      );
    }

    // hasMasterCV === null (unknown)
    return (
      <Button type="default" onClick={() => navigate('/candidate/cv')} block size="large">
        Evaluate Fit (Upload CV)
      </Button>
    );
  };

  const skillTags = Array.isArray(position.skills) ? position.skills : [];
  const minimumFitScore = position.minimumFitScore != null ? Math.round(position.minimumFitScore) : null;
  const hasAnalysis = applicationData?.overallStatus != null;
  const score = applicationData
    ? Math.round(((applicationData.technicalScore ?? 0) + (applicationData.experienceScore ?? 0)) / 2)
    : null;

  return (
    <Card
      hoverable
      style={{ borderRadius: 8, border: '1px solid #f0f0f0', height: '100%', display: 'flex', flexDirection: 'column' }}
      styles={{ body: { padding: '24px', display: 'flex', flexDirection: 'column', flex: 1 } }}
    >
      {/* Header */}
      <div style={{ marginBottom: 12 }}>
        <Text style={{ fontSize: 20, fontWeight: 500, textTransform: 'uppercase', letterSpacing: 1 }}>
          {position.seniority}
        </Text>
        <Title level={4} style={{ margin: '4px 0 0', color: '#1677ff', lineHeight: 1.2 }}>
          {position.title}
        </Title>
      </div>

      {/* Skills */}
      {skillTags.length > 0 && (
        <Space size={[4, 6]} wrap style={{ marginBottom: 16 }}>
          {skillTags.map((skill) => (
            <Tag key={skill} color="blue" style={{ margin: 0 }}>{skill}</Tag>
          ))}
        </Space>
      )}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <Text type="secondary" style={{ fontSize: 13 }}>
          <CalendarOutlined style={{ marginRight: 6 }} />
          Posted: {dayjs(position.openedAt).format('DD MMM YYYY')}
        </Text>
        {minimumFitScore != null && (
          <Tooltip title="Minimum fit score required for this position">
            <Tag color="volcano" style={{ margin: 0, fontWeight: 600 }}>
              Min. Score: {minimumFitScore}
            </Tag>
          </Tooltip>
        )}
      </div>

      {/* Application analysis block */}
      {applicationData && (
        <div style={{ padding: '12px', background: '#fafafa', borderRadius: 6, marginBottom: 16, border: '1px solid #f0f0f0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: hasAnalysis ? 8 : 0 }}>
            <Tag color={STAGE_CONFIG[applicationData.recruitmentStage]?.color || 'default'} style={{ margin: 0 }}>
              {STAGE_CONFIG[applicationData.recruitmentStage]?.label || applicationData.recruitmentStage}
            </Tag>
            {applicationData.recruitmentStage === 'INTERVIEW_SCHEDULED' && applicationData.interviewSchedule && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {dayjs(applicationData.interviewSchedule).format('DD MMM YYYY, HH:mm')}
              </Text>
            )}
          </div>

          {hasAnalysis && (
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text type="secondary" style={{ fontSize: 13 }}>Match</Text>
                <Badge
                  status={MATCH_STATUS_COLOR[applicationData.overallStatus] || 'default'}
                  text={<Text strong style={{ fontSize: 12 }}>{formatStatus(applicationData.overallStatus)}</Text>}
                />
              </div>
              {score !== null && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text type="secondary" style={{ fontSize: 13 }}>Score</Text>
                  <Space size={4}>
                    <Text strong style={{ color: score >= 70 ? '#52c41a' : '#cf1322' }}>{score} / 100</Text>
                    {applicationData.aiAssessment && (
                      <Tooltip title="View AI Analysis">
                        <InfoCircleOutlined
                          style={{ color: '#1677ff', cursor: 'pointer' }}
                          onClick={() => setIsAnalysisModalOpen(true)}
                        />
                      </Tooltip>
                    )}
                  </Space>
                </div>
              )}
            </Space>
          )}
        </div>
      )}

      <div style={{ flex: 1 }} />
      <Divider style={{ margin: '16px 0' }} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {renderAction()}
        <Button icon={<FileTextOutlined />} onClick={handleOpenJD} block>
          View Job Description
        </Button>
      </div>

      {/* AI Analysis Modal */}
      {applicationData?.aiAssessment && (
        <Modal
          title="AI Analysis"
          open={isAnalysisModalOpen}
          onCancel={() => setIsAnalysisModalOpen(false)}
          footer={[<Button key="close" onClick={() => setIsAnalysisModalOpen(false)}>Close</Button>]}
          width={600}
        >
          {/* Score summary */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Technical', value: applicationData.technicalScore },
              { label: 'Experience', value: applicationData.experienceScore },
              { label: 'Overall', value: score },
            ].map(({ label, value }) => (
              <div
                key={label}
                style={{ flex: 1, textAlign: 'center', padding: '12px 8px', background: '#fafafa', borderRadius: 8, border: '1px solid #f0f0f0' }}
              >
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>{label}</Text>
                <Text strong style={{ fontSize: 22, color: value != null && value >= 70 ? '#52c41a' : '#cf1322' }}>
                  {value ?? '-'}
                </Text>
                <Text type="secondary" style={{ fontSize: 12 }}> / 100</Text>
              </div>
            ))}
          </div>

          {score !== null && score < 70 ? (
            <LearningPathCard
              score={score}
              missingSkills={[]}
              learningPathText={applicationData.learningPath || applicationData.aiAssessment}
            />
          ) : (
            <div className="jd-markdown-body">
              <ReactMarkdown>{applicationData.aiAssessment}</ReactMarkdown>
            </div>
          )}
        </Modal>
      )}

      {/* Learning Path Modal (no application yet) */}
      <Modal
        title="Suggested Learning Path"
        open={isLearningPathModalOpen}
        onCancel={() => setIsLearningPathModalOpen(false)}
        footer={[<Button key="close" onClick={() => setIsLearningPathModalOpen(false)}>Close</Button>]}
        width={600}
      >
        <LearningPathCard score={0} missingSkills={[]} learningPathText="" />
      </Modal>

      {/* JD Modal */}
      <Modal
        title={`${position.seniority ? position.seniority + ' · ' : ''}${position.title}`}
        open={isJDModalOpen}
        onCancel={() => setIsJDModalOpen(false)}
        footer={[
          position.driveFileUrl && (
            <Button key="pdf" icon={<FilePdfOutlined />} onClick={() => window.open(position.driveFileUrl, '_blank')}>
              Open PDF
            </Button>
          ),
          <Button key="close" type="primary" onClick={() => setIsJDModalOpen(false)}>Close</Button>,
        ].filter(Boolean)}
        width={720}
      >
        {jdLoading ? (
          <div style={{ textAlign: 'center', padding: '40px' }}><Spin /></div>
        ) : (
          <div className="jd-markdown-body" style={{ maxHeight: '60vh', overflowY: 'auto', padding: '4px 0' }}>
            <ReactMarkdown>{jdText || 'No job description content available.'}</ReactMarkdown>
          </div>
        )}
      </Modal>
    </Card>
  );
};

export default PositionCard;
