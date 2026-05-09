import React, { useEffect } from 'react';
import { Modal, Form, Input, Select, Upload, Button } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;

const SENIORITY_OPTIONS = ['Intern', 'Fresher', 'Junior', 'Middle', 'Senior', 'Lead', 'Manager'];

const SKILL_PRESETS = [
  'Java', 'Spring Boot', 'Python', 'FastAPI', 'React', 'Vue', 'Angular',
  'NodeJS', 'Go', 'C#', '.NET', 'TypeScript', 'Docker', 'Kubernetes',
  'AWS', 'GCP', 'Azure', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis',
];

const PositionFormModal = ({ open, onCancel, onSave, initialData, loading }) => {
  const [form] = Form.useForm();
  const isEdit = !!initialData;

  useEffect(() => {
    if (open) {
      if (initialData) {
        form.setFieldsValue({
          title: initialData.title,
          seniority: initialData.seniority,
          skills: initialData.skills ?? [],
        });
      } else {
        form.resetFields();
      }
    }
  }, [open, initialData, form]);

  const uploadProps = {
    name: 'file',
    multiple: false,
    maxCount: 1,
    beforeUpload: () => false,
  };

  return (
    <Modal
      title={isEdit ? 'Edit Position' : 'Create New Position'}
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={loading}>
          Cancel
        </Button>,
        <Button key="submit" type="primary" onClick={() => form.submit()} loading={loading}>
          {isEdit ? 'Save Changes' : 'Create'}
        </Button>,
      ]}
      destroyOnClose
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={onSave}
        style={{ marginTop: 16 }}
      >
        <Form.Item
          name="title"
          label="Job Title"
          rules={[{ required: true, message: 'Please enter job title' }]}
        >
          <Input placeholder="e.g. Senior Java Developer" />
        </Form.Item>

        <Form.Item
          name="seniority"
          label="Seniority Level"
          rules={[{ required: true, message: 'Please select seniority' }]}
        >
          <Select placeholder="Select seniority">
            {SENIORITY_OPTIONS.map((s) => (
              <Select.Option key={s} value={s}>
                {s}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          name="skills"
          label="Required Skills"
          rules={[{ required: true, message: 'Please add at least one skill' }]}
        >
          <Select
            mode="tags"
            placeholder="Type or select skills (e.g. Java, React)"
            options={SKILL_PRESETS.map((s) => ({ value: s, label: s }))}
            tokenSeparators={[',']}
          />
        </Form.Item>

        {!isEdit && (
          <Form.Item
            name="file"
            label="Job Description File (JD)"
            valuePropName="fileList"
            getValueFromEvent={(e) => (Array.isArray(e) ? e : e?.fileList)}
            rules={[{ required: true, message: 'Please upload JD file' }]}
          >
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">Click or drag JD file here to upload</p>
            </Dragger>
          </Form.Item>
        )}
      </Form>
    </Modal>
  );
};

export default PositionFormModal;
