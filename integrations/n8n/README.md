# n8n Workflow Templates for TrustRAG

Three ready-to-import workflows demonstrating trust-gated automation.

## Workflows

1. **[Doc Ingestion](workflows/doc-ingestion.json)** -- Auto-upload PDFs from Google Drive to TrustRAG, Slack notify on success/failure
2. **[Slack Ask with Trust Gate](workflows/slack-ask-trust-gate.json)** -- `/ask` Slack command with trust-filtered replies; low-trust answers route to `#review-queue`
3. **[Daily Low-Confidence Digest](workflows/daily-low-confidence-digest.json)** -- Daily 7 AM digest of queries with trust < 60, sent via Slack and email

## Prerequisites

- [n8n](https://n8n.io/) v1.0+ (self-hosted or cloud)
- A running TrustRAG backend instance
- Slack workspace with a bot token (for workflows 1, 2, 3)
- Google Drive OAuth credentials (for workflow 1)
- SMTP credentials (for workflow 3 email delivery)

## Import Instructions

1. Open n8n -> **Workflows** -> **Import from File**
2. Select the `.json` file for the workflow you want
3. Configure credentials:
   - Set environment variable `TRUSTRAG_BACKEND_URL` (e.g., `http://trustrag.local:8000`)
   - Connect Slack Bot token via n8n credential setup
   - Connect Google Drive OAuth via n8n credential setup (workflow 1)
   - Connect SMTP account via n8n credential setup (workflow 3)
4. **Activate** the workflow

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `TRUSTRAG_BACKEND_URL` | All | Base URL of TrustRAG backend (e.g., `http://localhost:8000`) |
| `SMTP_FROM` | Workflow 3 | Sender email for digest (default: `trustrag@example.com`) |
| `ADMIN_EMAIL` | Workflow 3 | Recipient email for digest (default: `admin@example.com`) |

## Workflow Details

### 1. Doc Ingestion

Watches a Google Drive folder for new files. When a PDF is detected:
- Uploads it to TrustRAG via `POST /api/documents/upload`
- Posts success notification to `#docs-indexed` Slack channel
- Posts error alert to `#errors` channel on failure
- Skips non-PDF files with a notification

**Required credentials:** Google Drive OAuth, Slack Bot Token

### 2. Slack Ask with Trust Gate

Receives `/ask` slash commands from Slack via webhook:
- Forwards the question to TrustRAG via `POST /api/query`
- If `trust_score >= 70`: replies directly to the user with answer + sources
- If `trust_score < 70`: acknowledges to user, posts full details to `#review-queue` for human review

**Required credentials:** Slack Bot Token

**Slack setup:** Register the n8n webhook URL as the slash command endpoint for `/ask` in your Slack app configuration.

### 3. Daily Low-Confidence Digest

Runs on a cron schedule (daily at 7:00 AM):
- Fetches audit entries with `trust < 60` from the last 24 hours via `GET /api/audit?max_trust=60&since_hours=24`
- Formats a Markdown table with time, trust score, question, and answer preview
- Posts to `#daily-digest` Slack channel AND sends via email
- Skips if no low-confidence entries exist

**Required credentials:** Slack Bot Token, SMTP Account

## Screenshots

See `docs/screenshots/` in the main TrustRAG repo:
- `n8n-doc-ingestion.png` -- Doc ingestion workflow canvas
- `n8n-slack-ask.png` -- Slack ask with trust gate canvas
- `n8n-daily-digest.png` -- Daily digest workflow canvas

## Customization Tips

- **Adjust trust threshold**: In the Slack Ask workflow, change the `70` in the IF node to your preferred threshold
- **Swap file source**: Replace Google Drive trigger with S3, Dropbox, or local file watch
- **Swap notification channel**: Replace Slack nodes with Microsoft Teams, Discord, or email
- **Change digest schedule**: Modify the cron trigger in workflow 3 (e.g., hourly, weekly)
- **Filter by date range**: Adjust `since_hours` parameter in the audit API call

## Troubleshooting

- **Upload fails with 413**: Increase backend's `max_file_size` setting or nginx `client_max_body_size`
- **Trust always low**: Ensure TrustRAG has indexed relevant documents for your queries
- **Slack doesn't receive**: Verify bot OAuth scope includes `chat:write` and bot is invited to the channel
- **Webhook not firing**: Check n8n workflow is **activated** (not just saved)
- **Email not sending**: Verify SMTP credentials and check spam folder
