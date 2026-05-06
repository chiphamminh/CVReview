import React from 'react';
import { Modal, Button } from 'antd';
import { ExclamationCircleOutlined } from '@ant-design/icons';

const DeleteWarningPopup = ({ 
  open, 
  onCancel, 
  onConfirm, 
  title = "Xác nhận xóa", 
  content = "Bạn có chắc chắn muốn thực hiện hành động này không? Hành động này không thể hoàn tác.",
  loading = false,
  confirmText = "Xóa",
  cancelText = "Hủy bỏ"
}) => {
  return (
    <Modal
      open={open}
      onCancel={onCancel}
      closable={false}
      footer={[
        <Button key="back" onClick={onCancel} disabled={loading}>
          {cancelText}
        </Button>,
        <Button key="submit" type="primary" danger loading={loading} onClick={onConfirm}>
          {confirmText}
        </Button>,
      ]}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
        <ExclamationCircleOutlined style={{ color: '#faad14', fontSize: '22px', marginTop: '4px' }} />
        <div>
          <h3 style={{ margin: '0 0 8px 0', fontSize: '16px' }}>{title}</h3>
          <p style={{ margin: 0, color: '#595959' }}>{content}</p>
        </div>
      </div>
    </Modal>
  );
};

export default DeleteWarningPopup;
