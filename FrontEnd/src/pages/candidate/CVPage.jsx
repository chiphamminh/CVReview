import { useState, useEffect } from 'react';
import { Typography, message } from 'antd';
import CVCard from '@/components/candidate/CVCard';
import WarningUpdateCVModal from '@/components/modals/WarningUpdateCVModal';
import UploadCVModal from '@/components/modals/UploadCVModal';
import UpdateCVModal from '@/components/modals/UpdateCVModal';
import DeleteWarningPopup from '@/components/modals/DeleteWarningPopup';
import useAuthStore from '@/store/authStore';
import { fetchCandidates } from '@/api/mockData';

const { Title, Paragraph } = Typography;

const CVPage = () => {
  const { user } = useAuthStore();
  const [cvData, setCvData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const [isWarningModalOpen, setIsWarningModalOpen] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    // Mock fetch user's Master CV
    const loadCV = async () => {
      try {
        const candidates = await fetchCandidates();
        const masterCv = candidates.find(c => c.type === 'EXTERNAL' && !c.position_id);
        
        if (masterCv) {
          masterCv.applications = [
            { positionName: 'Senior Frontend Developer', stage: 'APPLIED', date: new Date().toISOString() }
          ];
        }
        
        setCvData(masterCv || null);
      } catch (error) {
        message.error('Failed to load CV');
      } finally {
        setLoading(false);
      }
    };
    loadCV();
  }, [user]);

  const handleUpdateClick = () => {
    if (cvData) {
      setIsWarningModalOpen(true);
    } else {
      setIsUploadModalOpen(true);
    }
  };

  const handleDeleteConfirm = () => {
    setCvData(null);
    setIsDeleteModalOpen(false);
    message.success('Master CV deleted successfully');
  };

  const handleWarningOk = () => {
    setIsWarningModalOpen(false);
    setIsUploadModalOpen(true);
  };

  const handleUploadModalCancel = () => {
    setIsUploadModalOpen(false);
    if (!cvData) {
      message.success('New Master CV uploaded successfully.');
      setCvData({
        name: user?.name || 'Candidate',
        email: user?.email || 'candidate@example.com',
        updatedAt: new Date().toISOString(),
        cvStatus: 'PARSED',
        driveFileUrl: '#',
        applications: []
      });
    }
  };

  const handleEditSave = (values) => {
    setCvData(prev => ({
      ...prev,
      name: values.name,
      email: values.email
    }));
    setIsEditModalOpen(false);
    message.success('CV Information updated successfully');
  };

  return (
    <div>
      <Title level={2}>My CV Management</Title>
      <Paragraph type="secondary">
        Manage your Master CV here. This CV will be used when you apply for positions via our AI Chatbot.
      </Paragraph>
      
      <div style={{ marginTop: '24px' }}>
        <CVCard 
          cvData={cvData} 
          onUpdateClick={handleUpdateClick} 
          onDeleteClick={() => setIsDeleteModalOpen(true)} 
          onEditClick={() => setIsEditModalOpen(true)}
        />
      </div>

      <WarningUpdateCVModal
        open={isWarningModalOpen}
        loading={false}
        onOk={handleWarningOk}
        onCancel={() => setIsWarningModalOpen(false)}
      />

      <UploadCVModal 
        open={isUploadModalOpen} 
        onCancel={handleUploadModalCancel} 
        positionName="Master Profile"
      />

      <UpdateCVModal
        title="Update CV Information"
        open={isEditModalOpen}
        onCancel={() => setIsEditModalOpen(false)}
        onSave={handleEditSave}
        initialData={cvData}
      />

      <DeleteWarningPopup
        open={isDeleteModalOpen}
        onCancel={() => setIsDeleteModalOpen(false)}
        onConfirm={handleDeleteConfirm}
        title="Delete Master CV"
        content="Are you sure you want to delete your Master CV? You will not be able to apply to any new positions until you upload a new one. This action cannot be undone."
      />
    </div>
  );
};

export default CVPage;
