import { useEffect, useState } from 'react';
import { Typography, Row, Col, Spin, Divider, Layout } from 'antd';
import { fetchActivePositions } from '@/api/mockData';
import PositionCard from '@/components/candidate/PositionCard';
import useAuthStore from '@/store/authStore';

const { Title, Paragraph } = Typography;
const { Content } = Layout;

const CareerPage = () => {
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchActivePositions();
        setPositions(data);
      } catch (error) {
        console.error('Failed to load positions', error);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  return (
    <div>
      {/* Hero Section */}
      <div style={{ textAlign: 'center', padding: '60px 0', background: 'linear-gradient(90deg, rgba(22,119,255,0.1) 0%, rgba(255,255,255,1) 100%)', borderRadius: '8px', marginBottom: '40px' }}>
        <Title style={{ fontSize: '48px', color: '#1677ff' }}>Join TechCorp</Title>
        <Paragraph style={{ fontSize: '18px', maxWidth: '600px', margin: '0 auto', color: '#595959' }}>
          We are a leading tech company building the future of AI-driven recruitment. 
          Discover your next career move and grow with us.
        </Paragraph>
      </div>

      {/* About Us & What We Do Section */}
      <Row gutter={[48, 48]} style={{ marginBottom: '60px' }}>
        <Col xs={24} md={12}>
          <Title level={3}>About Us</Title>
          <Paragraph style={{ fontSize: '16px', color: '#595959' }}>
            At TechCorp, we believe that the right talent can transform an organization. 
            Founded in 2020, we have been bridging the gap between exceptional individuals 
            and innovative companies. Our core values revolve around transparency, 
            continuous learning, and pushing the boundaries of what's possible with technology.
          </Paragraph>
        </Col>
        <Col xs={24} md={12}>
          <Title level={3}>What We Do</Title>
          <Paragraph style={{ fontSize: '16px', color: '#595959' }}>
            We build scalable, intelligent platforms that simplify complex workflows. 
            Our flagship product uses advanced AI and RAG pipelines to match candidates 
            with their dream jobs, ensuring a fair, skills-based evaluation process. 
            When you join us, you're not just taking a job; you're joining a mission.
          </Paragraph>
        </Col>
      </Row>

      <Divider />

      {/* Active Positions Section */}
      <div id="open-positions" style={{ padding: '20px 0' }}>
        <Title level={2} style={{ marginBottom: '32px' }}>Open Positions</Title>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
          </div>
        ) : (
          <div>
            {positions.length === 0 ? (
              <Paragraph>No active positions at the moment. Please check back later.</Paragraph>
            ) : (
              <Row gutter={[24, 24]} align="stretch">
                {positions.map(pos => (
                  <Col xs={24} sm={12} lg={8} key={pos.id} style={{ display: 'flex' }}>
                    <div style={{ width: '100%' }}>
                      <PositionCard position={pos} />
                    </div>
                  </Col>
                ))}
              </Row>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CareerPage;
