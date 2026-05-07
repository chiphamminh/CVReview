import React, { useEffect } from 'react';
import { Modal, Form, Input, DatePicker, Button, Typography, InputNumber } from 'antd';

const { Text } = Typography;

const SendOfferModal = ({ open, onCancel, onSave, candidateData }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (open) {
      form.resetFields();
    }
  }, [open, candidateData, form]);

  const handleFinish = (values) => {
    onSave({
      ...values,
      startDate: values.startDate.toISOString(),
      expirationDate: values.expirationDate?.toISOString(),
    });
  };

  return (
    <Modal
      title="Send Offer Letter"
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>Cancel</Button>,
        <Button key="submit" type="primary" onClick={() => form.submit()}>
          Send Offer
        </Button>
      ]}
      destroyOnClose
    >
      <div style={{ marginBottom: 24, padding: 12, background: '#f5f5f5', borderRadius: 8 }}>
        <p style={{ margin: '0 0 8px 0' }}><Text type="secondary">Candidate:</Text> <strong>{candidateData?.name}</strong></p>
        <p style={{ margin: 0 }}><Text type="secondary">Email:</Text> {candidateData?.email}</p>
      </div>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
      >
        <Form.Item
          name="salary"
          label="Offered Salary ($)"
          rules={[{ required: true, message: 'Please enter salary' }]}
        >
          <InputNumber
            style={{ width: '100%' }}
            formatter={(value) => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
            parser={(value) => value?.replace(/\$\s?|(,*)/g, '')}
            min={0}
          />
        </Form.Item>

        <Form.Item
          name="benefits"
          label="Benefits & Perks"
          rules={[{ required: true, message: 'Please enter benefits' }]}
        >
          <Input.TextArea 
            rows={3} 
            placeholder="E.g. Health insurance, 13th month salary, hybrid work..." 
          />
        </Form.Item>

        <Form.Item
          name="startDate"
          label="Expected Start Date"
          rules={[{ required: true, message: 'Please select start date' }]}
        >
          <DatePicker style={{ width: '100%' }} format="YYYY-MM-DD" />
        </Form.Item>

        <Form.Item
          name="expirationDate"
          label="Offer Expiration Date"
          rules={[{ required: true, message: 'Please select expiration date' }]}
        >
          <DatePicker style={{ width: '100%' }} format="YYYY-MM-DD" />
        </Form.Item>

        <Form.Item
          name="note"
          label="Additional Notes"
        >
          <Input.TextArea 
            rows={2} 
            placeholder="Internal notes or messages to candidate..." 
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default SendOfferModal;
