import { Modal, Typography, Alert } from 'antd';
import { ExclamationCircleOutlined } from '@ant-design/icons';

const { Text, Paragraph } = Typography;

const WarningUpdateCVModal = ({ open, onOk, onCancel, loading }) => {
  return (
    <Modal
      title={
        <span>
          <ExclamationCircleOutlined style={{ color: '#faad14', marginRight: 8 }} />
          Warning: Updating Master CV
        </span>
      }
      open={open}
      onOk={onOk}
      onCancel={onCancel}
      confirmLoading={loading}
      okText="Yes, Update CV"
      cancelText="Cancel"
      okButtonProps={{ danger: true }}
    >
      <Alert
        message="Critical Action"
        description="Uploading a new Master CV will affect your existing applications."
        type="warning"
        showIcon
        style={{ marginBottom: 16 }}
      />
      <Paragraph>
        According to our system policy, if you upload a new Master CV:
      </Paragraph>
      <ul>
        <li>
          <Text strong type="danger">All your previous Application CVs and Analysis records will be soft-deleted.</Text>
        </li>
        <li>
          You will need to <Text strong>re-apply</Text> for any positions you are currently applying to using the new CV.
        </li>
      </ul>
      <Paragraph>
        Are you sure you want to proceed?
      </Paragraph>
    </Modal>
  );
};

export default WarningUpdateCVModal;
