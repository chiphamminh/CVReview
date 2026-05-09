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
            colorPrimary: '#1677ff',
            borderRadius: 8,
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
