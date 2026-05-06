import React from 'react';

const FilePreview = ({ fileUrl, title = 'Bản xem trước tài liệu' }) => {
  if (!fileUrl) {
    return <div style={{ textAlign: 'center', padding: '2rem' }}>Không có tài liệu nào để hiển thị.</div>;
  }

  // Chuyển đổi link Google Drive thành dạng nhúng (embed)
  // Nếu url gốc là dạng /view?usp=sharing, ta cần đổi nó thành /preview
  const embedUrl = fileUrl.replace('/view?usp=sharing', '/preview').replace('/view', '/preview');

  return (
    <div style={{ width: '100%', height: '70vh', borderRadius: '8px', overflow: 'hidden', border: '1px solid #d9d9d9' }}>
      <iframe
        title={title}
        src={embedUrl}
        width="100%"
        height="100%"
        frameBorder="0"
        allow="autoplay"
      ></iframe>
    </div>
  );
};

export default FilePreview;
