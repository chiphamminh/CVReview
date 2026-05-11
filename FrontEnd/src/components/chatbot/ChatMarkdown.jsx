import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const tableStyle = {
  borderCollapse: 'collapse',
  width: '100%',
  marginBottom: 8,
  fontSize: 13,
};

const thStyle = {
  border: '1px solid #d9d9d9',
  padding: '6px 12px',
  background: '#fafafa',
  fontWeight: 600,
  textAlign: 'left',
  whiteSpace: 'nowrap',
};

const tdStyle = {
  border: '1px solid #d9d9d9',
  padding: '6px 12px',
  verticalAlign: 'top',
};

const components = {
  table: ({ node, ...props }) => (
    <div style={{ overflowX: 'auto', marginBottom: 8 }}>
      <table style={tableStyle} {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead {...props} />,
  tbody: ({ node, ...props }) => <tbody {...props} />,
  tr: ({ node, ...props }) => <tr {...props} />,
  th: ({ node, ...props }) => <th style={thStyle} {...props} />,
  td: ({ node, ...props }) => <td style={tdStyle} {...props} />,
};

const ChatMarkdown = ({ children }) => (
  <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
    {children}
  </ReactMarkdown>
);

export default ChatMarkdown;
