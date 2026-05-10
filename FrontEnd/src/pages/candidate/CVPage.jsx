import { useState, useEffect, useCallback } from 'react';
import { Typography, Button, App } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import CVCard from '@/components/candidate/CVCard';
import CandidateUploadCVModal from '@/components/modals/CandidateUploadCVModal';
import UpdateCVModal from '@/components/modals/UpdateCVModal';
import DeleteWarningPopup from '@/components/modals/DeleteWarningPopup';
import useAuthStore from '@/store/authStore';
import { candidateApi } from '@/api/candidate.api';

const { Title, Paragraph } = Typography;

const CVPage = () => {
  const { message } = App.useApp();
  const { setHasMasterCV } = useAuthStore();

  const [cvData, setCvData] = useState(null);
  const [applications, setApplications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [cvRes, appRes] = await Promise.allSettled([
        candidateApi.getMyCV(),
        candidateApi.getMyApplications(),
      ]);

      const cv = cvRes.status === 'fulfilled' ? (cvRes.value.data || null) : null;
      setCvData(cv);
      setHasMasterCV(!!cv);

      if (appRes.status === 'fulfilled') {
        setApplications(appRes.value.data || []);
      }
    } catch {
      message.error('Failed to load CV data');
    } finally {
      setLoading(false);
    }
  }, [message, setHasMasterCV]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleUploadSuccess = () => {
    setIsUploadModalOpen(false);
    fetchData();
  };

  const handleDeleteConfirm = async () => {
    setIsDeleting(true);
    try {
      await candidateApi.deleteMany([cvData.cvId]);
      setCvData(null);
      setApplications([]);
      setHasMasterCV(false);
      setIsDeleteModalOpen(false);
      message.success('Master CV deleted successfully');
    } catch (err) {
      message.error(err.response?.data?.message || 'Failed to delete CV');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleEditSave = async (values) => {
    setIsSaving(true);
    try {
      await candidateApi.updateInfo(cvData.cvId, { name: values.name, email: values.email });
      setCvData((prev) => ({ ...prev, name: values.name, email: values.email }));
      setIsEditModalOpen(false);
      message.success('CV information updated successfully');
    } catch (err) {
      message.error(err.response?.data?.message || 'Failed to update CV');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <Title level={2} style={{ margin: 0 }}>My CV Management</Title>
          <Paragraph type="secondary" style={{ marginTop: 4 }}>
            Manage your Master CV. This CV is used when you apply for positions via the AI Chatbot.
          </Paragraph>
        </div>
        {!cvData && (
          <Button
            type="primary"
            icon={<UploadOutlined />}
            onClick={() => setIsUploadModalOpen(true)}
          >
            Upload CV
          </Button>
        )}
      </div>

      <div style={{ marginTop: '24px' }}>
        <CVCard
          cvData={cvData}
          loading={loading}
          applications={applications}
          onDeleteClick={() => setIsDeleteModalOpen(true)}
          onEditClick={() => setIsEditModalOpen(true)}
        />
      </div>

      <CandidateUploadCVModal
        open={isUploadModalOpen}
        onCancel={() => setIsUploadModalOpen(false)}
        onSuccess={handleUploadSuccess}
        isReupload={!!cvData}
      />

      <UpdateCVModal
        title="Update CV Information"
        open={isEditModalOpen}
        onCancel={() => setIsEditModalOpen(false)}
        onSave={handleEditSave}
        initialData={cvData}
        loading={isSaving}
      />

      <DeleteWarningPopup
        open={isDeleteModalOpen}
        onCancel={() => setIsDeleteModalOpen(false)}
        onConfirm={handleDeleteConfirm}
        loading={isDeleting}
        title="Delete Master CV"
        content="Are you sure you want to delete your Master CV? All active applications will be invalidated. This action cannot be undone."
      />
    </div>
  );
};

export default CVPage;
