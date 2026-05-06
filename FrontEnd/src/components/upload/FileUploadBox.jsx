import React from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;

const FileUploadBox = ({ onFileChange, accept = '.pdf,.doc,.docx', maxCount = 1 }) => {
  const props = {
    name: 'file',
    multiple: maxCount > 1,
    accept,
    maxCount,
    beforeUpload: (file) => {
      // Return false so we handle upload manually instead of antd doing it automatically
      if (onFileChange) {
        onFileChange(file);
      }
      return false;
    },
    onChange(info) {
      const { status } = info.file;
      if (status !== 'uploading') {
        // console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  return (
    <Dragger {...props}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">Nhấp hoặc kéo thả file vào khu vực này để tải lên</p>
      <p className="ant-upload-hint">
        Hỗ trợ các định dạng tải lên: {accept}. Kích thước tối đa 5MB.
      </p>
    </Dragger>
  );
};

export default FileUploadBox;
