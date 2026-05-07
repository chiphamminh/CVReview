import React, { useEffect } from 'react';
import { Modal, Form, Input, Select, Upload, Button, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;
const { Option } = Select;

const PositionFormModal = ({ open, onCancel, onSave, initialData }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (open) {
      if (initialData) {
        form.setFieldsValue({
          name: initialData.name,
          language: initialData.language,
          level: initialData.level,
        });
      } else {
        form.resetFields();
      }
    }
  }, [open, initialData, form]);

  const handleFinish = (values) => {
    // values will contain name, language, level, and possibly file
    onSave(values);
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    maxCount: 1,
    beforeUpload: () => false, // manual upload
  };

  return (
    <Modal
      title={initialData ? "Edit Position" : "Create New Position"}
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>Cancel</Button>,
        <Button key="submit" type="primary" onClick={() => form.submit()}>
          {initialData ? "Save Changes" : "Create"}
        </Button>
      ]}
      destroyOnClose
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        style={{ marginTop: '16px' }}
      >
        <Form.Item
          name="name"
          label="Job Title"
          rules={[{ required: true, message: 'Please enter job title' }]}
        >
          <Input placeholder="e.g. Senior Java Developer" />
        </Form.Item>

        <Form.Item
          name="language"
          label="Primary Language / Tech Stack"
          rules={[{ required: true, message: 'Please select or enter language' }]}
        >
          <Select placeholder="Select language">
            <Option value="Java">Java</Option>
            <Option value="Python">Python</Option>
            <Option value="React">React</Option>
            <Option value="NodeJS">NodeJS</Option>
            <Option value="Go">Go</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="level"
          label="Experience Level"
          rules={[{ required: true, message: 'Please select level' }]}
        >
          <Select placeholder="Select level">
            <Option value="Junior">Junior</Option>
            <Option value="Middle">Middle</Option>
            <Option value="Senior">Senior</Option>
            <Option value="Lead">Lead</Option>
            <Option value="Manager">Manager</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="file"
          label="Job Description File (JD)"
          valuePropName="fileList"
          getValueFromEvent={(e) => {
            if (Array.isArray(e)) return e;
            return e?.fileList;
          }}
          rules={[{ required: !initialData, message: 'Please upload JD file' }]}
        >
          <Dragger {...uploadProps}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">Click or drag file to this area to upload</p>
          </Dragger>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default PositionFormModal;
