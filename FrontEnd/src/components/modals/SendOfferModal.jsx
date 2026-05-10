import React, { useEffect, useState } from 'react';
import { Modal, Form, DatePicker, Button, Typography, Upload } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';

const { Text } = Typography;

const SendOfferModal = ({ open, onCancel, onSave, candidateData, loading }) => {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState([]);

  useEffect(() => {
    if (open) {
      form.resetFields();
      setFileList([]);
    }
  }, [open, candidateData, form]);

  const handleFinish = (values) => {
    onSave({
      startDate: values.startDate.format('YYYY-MM-DD'),
      expirationDate: values.expirationDate?.format('YYYY-MM-DD'),
      files: fileList,
    });
  };

  return (
    <Modal
      title="Send Offer Letter"
      open={open}
      onCancel={onCancel}
      width={520}
      footer={[
        <Button key="cancel" onClick={onCancel}>Cancel</Button>,
        <Button key="submit" type="primary" loading={loading} onClick={() => form.submit()}>
          Send Offer
        </Button>,
      ]}
      destroyOnHidden
    >
      <div style={{ marginBottom: 24, padding: 12, background: '#f5f5f5', borderRadius: 8 }}>
        <p style={{ margin: '0 0 8px 0' }}><Text type="secondary">Candidate:</Text> <strong>{candidateData?.name}</strong></p>
        <p style={{ margin: 0 }}><Text type="secondary">Email:</Text> {candidateData?.email}</p>
      </div>

      <Form form={form} layout="vertical" onFinish={handleFinish}>
        <Form.Item
          name="startDate"
          label="Expected Start Date"
          rules={[{ required: true, message: 'Please select start date' }]}
        >
          <DatePicker
            style={{ width: '100%' }}
            format="YYYY-MM-DD"
            disabledDate={(current) => current && current < dayjs().startOf('day')}
          />
        </Form.Item>

        <Form.Item
          name="expirationDate"
          label="Offer Expiration Date"
          rules={[{ required: true, message: 'Please select expiration date' }]}
        >
          <DatePicker
            style={{ width: '100%' }}
            format="YYYY-MM-DD"
            disabledDate={(current) => current && current < dayjs().startOf('day')}
          />
        </Form.Item>

        <Form.Item label="Attachments">
          <Upload
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList: newList }) => setFileList(newList)}
            multiple
          >
            <Button icon={<UploadOutlined />}>Select Files</Button>
          </Upload>
          <Text type="secondary" style={{ fontSize: 12, marginTop: 4, display: 'block' }}>
            Labor contract, training guide, etc.
          </Text>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default SendOfferModal;
