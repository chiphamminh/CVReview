import React from 'react';
import { Modal, Button } from 'antd';
import FileUploadBox from '@/components/upload/FileUploadBox';

const UploadPopup = ({ open, onCancel, onUpload, title = "Tải lên tài liệu", loading = false }) => {
  const [file, setFile] = React.useState(null);

  const handleFileChange = (selectedFile) => {
    setFile(selectedFile);
  };

  const handleOk = () => {
    if (file && onUpload) {
      onUpload(file);
    }
  };

  return (
    <Modal
      title={title}
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="back" onClick={onCancel}>
          Hủy bỏ
        </Button>,
        <Button key="submit" type="primary" loading={loading} onClick={handleOk} disabled={!file}>
          Tải lên
        </Button>,
      ]}
      destroyOnClose
    >
      <div style={{ marginTop: 24, marginBottom: 24 }}>
        <FileUploadBox onFileChange={handleFileChange} />
      </div>
    </Modal>
  );
};

export default UploadPopup;
