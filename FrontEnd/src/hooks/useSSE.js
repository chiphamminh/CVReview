import { useState, useEffect, useRef } from 'react';

const useSSE = (url, options = {}) => {
  const { onMessage, onError, onOpen, enabled = true } = options;
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    if (!enabled || !url) return;

    // Khởi tạo EventSource
    eventSourceRef.current = new EventSource(url);

    eventSourceRef.current.onopen = (e) => {
      setIsConnected(true);
      if (onOpen) onOpen(e);
    };

    eventSourceRef.current.onmessage = (e) => {
      // Cố gắng parse JSON nếu có thể
      let data = e.data;
      try {
        data = JSON.parse(e.data);
      } catch (err) {
        // Data là chuỗi thuần
      }
      
      if (onMessage) onMessage(data, e);
    };

    eventSourceRef.current.onerror = (e) => {
      setIsConnected(false);
      if (onError) onError(e);
      // Đóng kết nối nếu có lỗi nghiêm trọng hoặc tùy chỉnh logic reconnect
      eventSourceRef.current.close();
    };

    // Cleanup khi component unmount hoặc url thay đổi
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        setIsConnected(false);
      }
    };
  }, [url, enabled]); // Chú ý: bỏ qua onMessage, onError để tránh re-render liên tục nếu là inline function

  const closeConnection = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      setIsConnected(false);
    }
  };

  return { isConnected, closeConnection };
};

export default useSSE;
