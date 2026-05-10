import React, { useEffect } from 'react';
import { Modal, Form, Input, DatePicker, Button, Typography } from 'antd';
import dayjs from 'dayjs';

const { Text } = Typography;

const ScheduleInterviewModal = ({ open, onCancel, onSave, candidateData }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (open) {
      if (candidateData?.interviewSchedule) {
        form.setFieldsValue({
          date: dayjs(candidateData.interviewSchedule),
          note: ''
        });
      } else {
        form.resetFields();
      }
    }
  }, [open, candidateData, form]);

  const handleFinish = (values) => {
    onSave({
      date: values.date.format('YYYY-MM-DDTHH:mm:ss'),
      note: values.note
    });
  };

  return (
    <Modal
      title={candidateData?.interviewSchedule ? "Re-schedule Interview" : "Schedule Interview"}
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>Cancel</Button>,
        <Button key="submit" type="primary" onClick={() => form.submit()}>
          Confirm Schedule
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
          name="date"
          label="Interview Date & Time"
          rules={[{ required: true, message: 'Please select date and time' }]}
        >
          <DatePicker
            showTime
            format="YYYY-MM-DD HH:mm"
            style={{ width: '100%' }}
            disabledDate={(current) => current && current < dayjs().startOf('day')}
          />
        </Form.Item>

        <Form.Item
          name="note"
          label="Notes (Optional)"
        >
          <Input.TextArea 
            rows={4} 
            placeholder="E.g. Technical interview with Team Lead. Google Meet link..." 
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default ScheduleInterviewModal;
