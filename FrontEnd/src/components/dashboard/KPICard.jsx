import { Card } from 'antd';

const KPICard = ({ icon, iconColor, title, value, suffix, note, loading, precision = 0 }) => {
  const formatted =
    typeof value === 'number'
      ? value.toLocaleString(undefined, {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        })
      : value;

  return (
    <Card loading={loading} styles={{ body: { padding: '20px 24px' } }} style={{ width: '100%', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
        <div
          style={{
            background: `${iconColor}1a`,
            color: iconColor,
            borderRadius: 10,
            padding: '10px 11px',
            fontSize: 22,
            lineHeight: 1,
            flexShrink: 0,
          }}
        >
          {icon}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, color: '#8c8c8c', marginBottom: 6 }}>{title}</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: '#141414', lineHeight: 1.2 }}>
            {formatted}
            {suffix && (
              <span style={{ fontSize: 14, fontWeight: 400, color: '#595959', marginLeft: 6 }}>
                {suffix}
              </span>
            )}
          </div>
          {note && (
            <div style={{ fontSize: 12, color: '#bfbfbf', marginTop: 4 }}>{note}</div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default KPICard;
