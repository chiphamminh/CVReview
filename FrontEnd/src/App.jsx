import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppRoutes from '@/routes/AppRoutes';
import { ConfigProvider, App as AntdApp } from 'antd';

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
      <ConfigProvider
        theme={{
          token: {
            colorPrimary: '#4F46E5',
            borderRadius: 8,
            colorBgLayout: '#F8FAFC',
          },
        }}
      >
        <AntdApp>
          <AppRoutes />
        </AntdApp>
      </ConfigProvider>
    </QueryClientProvider>
  );
}

export default App;
