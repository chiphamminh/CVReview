import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import useAuthStore from '@/store/authStore';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

/**
 * Hook kết nối SSE với Authorization header (dùng fetch-event-source thay EventSource).
 *
 * @param {string|null} url  - Path tương đối (VD: /tracking/batch-123/stream). Truyền null để chưa kết nối.
 * @param {object} options
 * @param {function} options.onMessage   - Callback(data: object, eventName: string) mỗi khi nhận event
 * @param {function} options.onError     - Callback(err) khi có lỗi không thể recover
 * @param {function} options.onOpen      - Callback khi kết nối thành công
 * @param {function} options.onClose     - Callback khi stream đóng bình thường
 * @param {boolean}  options.enabled     - Bật/tắt kết nối (default true)
 */
const useSSE = (url, options = {}) => {
  const { onMessage, onError, onOpen, onClose, enabled = true } = options;
  const [isConnected, setIsConnected] = useState(false);
  const abortControllerRef = useRef(null);

  const closeConnection = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsConnected(false);
  }, []);

  useEffect(() => {
    if (!enabled || !url) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const token = useAuthStore.getState().token;

    fetchEventSource(`${BASE_URL}${url}`, {
      method: 'GET',
      headers: {
        Authorization: token ? `Bearer ${token}` : '',
        Accept: 'text/event-stream',
      },
      signal: controller.signal,

      onopen: async (response) => {
        if (response.ok) {
          setIsConnected(true);
          if (onOpen) onOpen(response);
        } else {
          throw new Error(`SSE open failed: ${response.status}`);
        }
      },

      onmessage: (event) => {
        let data = event.data;
        try {
          data = JSON.parse(event.data);
        } catch {
          // data là chuỗi thuần, giữ nguyên
        }
        if (onMessage) onMessage(data, event.event);
      },

      onerror: (err) => {
        // AbortError là đóng chủ động — không phải lỗi thực sự
        if (err?.name === 'AbortError') return;
        setIsConnected(false);
        if (onError) onError(err);
        // Trả về undefined để library không tự retry
        throw err;
      },

      onclose: () => {
        setIsConnected(false);
        if (onClose) onClose();
      },

      // Tắt auto-retry của library — logic reconnect do caller tự quyết
      openWhenHidden: true,
    });

    return () => {
      controller.abort();
      setIsConnected(false);
    };
  }, [url, enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  return { isConnected, closeConnection };
};

export default useSSE;
