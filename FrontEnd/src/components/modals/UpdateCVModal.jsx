import React, { useEffect } from 'react';
import { Modal, Form, Input, Button } from 'antd';

const UpdateCVModal = ({ open, onCancel, onSave, initialData }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (open && initialData) {
      form.setFieldsValue({
        name: initialData.name,
        email: initialData.email,
      });
    } else {
      form.resetFields();
    }
  }, [open, initialData, form]);

  const handleFinish = (values) => {
    onSave(values);
  };

  return (
    <Modal
      title="Update Candidate Info (Internal)"
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>Cancel</Button>,
        <Button key="submit" type="primary" onClick={() => form.submit()}>
          Save Changes
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
          label="Full Name"
          rules={[{ required: true, message: 'Please enter candidate name' }]}
        >
          <Input placeholder="Enter full name" />
        </Form.Item>

        <Form.Item
          name="email"
          label="Email Address"
          rules={[
            { required: true, message: 'Please enter email' },
            { type: 'email', message: 'Please enter a valid email' }
          ]}
        >
          <Input placeholder="Enter email address" />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default UpdateCVModal;
