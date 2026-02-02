# Monitoring Guide

## Accessing Metrics

### Prometheus Metrics

Visit: `https://your-api-url.onrender.com/metrics`

Key metrics to monitor:
- `fraud_predictions_total` - Total predictions made
- `fraud_detected_total` - Total fraud cases detected
- `prediction_duration_seconds` - Processing time
- `active_requests` - Current load

### API Statistics

Visit: `https://your-api-url.onrender.com/stats`

Returns:
- Total predictions
- Fraud detection rate
- Average processing time
- Model accuracy

## Setting Up Alerts

### Render Dashboard

1. Go to Render dashboard
2. Select your service
3. Click "Metrics" tab
4. View:
   - CPU usage
   - Memory usage
   - Request rate
   - Response time

### UptimeRobot (Free)

1. Sign up at https://uptimerobot.com
2. Add new monitor:
   - Type: HTTP(S)
   - URL: https://your-api-url.onrender.com/health
   - Interval: 5 minutes
3. Get alerts via email/SMS when down

## Logging

### View Logs
```bash
# View last 100 logs
tail -n 100 logs/predictions.json

# Monitor in real-time
tail -f logs/predictions.json
```

### Log Analysis
```python
import json
import pandas as pd

# Load logs
with open('logs/predictions.json') as f:
    logs = json.load(f)

df = pd.DataFrame(logs)

# Analyze fraud rate over time
print(df['prediction'].value_counts())

# Average processing time
print(f"Avg processing time: {df['processing_time_ms'].mean():.2f}ms")
```

## Daily Health Checks

The CI/CD pipeline runs daily at 2 AM UTC to:
- Test all endpoints
- Check model accuracy
- Verify deployment health
- Run performance tests