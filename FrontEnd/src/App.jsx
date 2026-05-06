import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppRoutes from '@/routes/AppRoutes';
import { ConfigProvider } from 'antd';

// Khởi tạo QueryClient
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider theme={{
        token: {
          colorPrimary: '#1677ff',
        },
      }}>
        <AppRoutes />
      </ConfigProvider>
    </QueryClientProvider>
  );
}

export default App;
