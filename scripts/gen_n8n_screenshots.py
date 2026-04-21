"""Generate placeholder n8n workflow canvas screenshots for documentation."""
from PIL import Image, ImageDraw, ImageFont
import os

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# n8n-inspired colors
BG = (36, 36, 36)
NODE_BG = (55, 55, 55)
NODE_BORDER = (120, 120, 120)
TRIGGER_COLOR = (255, 170, 51)  # orange
IF_COLOR = (255, 204, 0)       # yellow
HTTP_COLOR = (102, 187, 255)   # blue
SLACK_COLOR = (74, 154, 202)   # slack blue
CODE_COLOR = (168, 130, 255)   # purple
EMAIL_COLOR = (255, 130, 130)  # red-ish
NOOP_COLOR = (150, 150, 150)   # gray
WEBHOOK_COLOR = (0, 200, 150)  # teal
TEXT_COLOR = (255, 255, 255)
CONN_COLOR = (100, 100, 100)
HEADER_COLOR = (50, 50, 50)


def draw_node(draw, x, y, w, h, label, color, sublabel=None):
    # rounded rect approximation
    draw.rectangle([x, y, x + w, y + h], fill=NODE_BG, outline=color, width=2)
    # color bar on left
    draw.rectangle([x, y, x + 5, y + h], fill=color)
    # label
    draw.text((x + 14, y + 8), label, fill=TEXT_COLOR)
    if sublabel:
        draw.text((x + 14, y + 28), sublabel, fill=(180, 180, 180))


def draw_connection(draw, x1, y1, x2, y2, label=None):
    draw.line([(x1, y1), (x2, y2)], fill=CONN_COLOR, width=2)
    # arrowhead
    if label:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2 - 10
        draw.text((mx, my), label, fill=(200, 200, 200))


def draw_header(draw, width, title):
    draw.rectangle([0, 0, width, 50], fill=HEADER_COLOR)
    draw.text((20, 12), f"n8n — {title}", fill=TEXT_COLOR)
    draw.text((width - 200, 12), "TrustRAG Integration", fill=(140, 140, 140))


def gen_doc_ingestion():
    w, h = 1200, 600
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, w, "TrustRAG Doc Ingestion")

    nw, nh = 180, 50
    # Nodes
    draw_node(draw, 50, 250, nw, nh, "Google Drive Watch", TRIGGER_COLOR, "Trigger")
    draw_node(draw, 290, 250, nw, nh, "Is PDF?", IF_COLOR, "IF condition")
    draw_node(draw, 530, 180, nw, nh, "Upload to TrustRAG", HTTP_COLOR, "HTTP POST")
    draw_node(draw, 530, 350, nw, nh, "Slack Notify Skipped", SLACK_COLOR, "Slack")
    draw_node(draw, 770, 180, nw, nh, "Upload OK?", IF_COLOR, "IF condition")
    draw_node(draw, 980, 120, nw, nh, "Slack Notify Success", SLACK_COLOR, "Slack")
    draw_node(draw, 980, 260, nw, nh, "Slack Alert Error", SLACK_COLOR, "Slack")

    # Connections
    draw_connection(draw, 230, 275, 290, 275)
    draw_connection(draw, 470, 260, 530, 205, "true")
    draw_connection(draw, 470, 290, 530, 375, "false")
    draw_connection(draw, 710, 205, 770, 205)
    draw_connection(draw, 950, 195, 980, 145, "true")
    draw_connection(draw, 950, 215, 980, 285, "false")

    # Watermark
    draw.text((20, h - 30), "Placeholder — import doc-ingestion.json into n8n for live canvas", fill=(100, 100, 100))

    img.save(os.path.join(SCREENSHOTS_DIR, "n8n-doc-ingestion.png"))
    print("Created n8n-doc-ingestion.png")


def gen_slack_ask():
    w, h = 1200, 600
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, w, "TrustRAG Slack Ask with Trust Gate")

    nw, nh = 190, 50
    draw_node(draw, 50, 260, nw, nh, "Slack /ask Webhook", WEBHOOK_COLOR, "Webhook POST")
    draw_node(draw, 310, 260, nw, nh, "Query TrustRAG", HTTP_COLOR, "HTTP POST")
    draw_node(draw, 570, 260, nw, nh, "Trust >= 70?", IF_COLOR, "IF condition")
    draw_node(draw, 830, 160, nw, nh, "Reply High Trust", WEBHOOK_COLOR, "Respond")
    draw_node(draw, 830, 360, nw, nh, "Reply Low Trust Ack", WEBHOOK_COLOR, "Respond")
    draw_node(draw, 1000, 440, nw, nh, "Post to Review Queue", SLACK_COLOR, "Slack")

    draw_connection(draw, 240, 285, 310, 285)
    draw_connection(draw, 500, 285, 570, 285)
    draw_connection(draw, 760, 270, 830, 185, "true")
    draw_connection(draw, 760, 300, 830, 385, "false")
    draw_connection(draw, 1020, 410, 1000, 465)

    draw.text((20, h - 30), "Placeholder — import slack-ask-trust-gate.json into n8n for live canvas", fill=(100, 100, 100))

    img.save(os.path.join(SCREENSHOTS_DIR, "n8n-slack-ask.png"))
    print("Created n8n-slack-ask.png")


def gen_daily_digest():
    w, h = 1200, 600
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, w, "TrustRAG Daily Low-Confidence Digest")

    nw, nh = 200, 50
    draw_node(draw, 50, 260, nw, nh, "Daily 7AM Trigger", TRIGGER_COLOR, "Schedule")
    draw_node(draw, 310, 260, nw, nh, "Fetch Low-Trust Entries", HTTP_COLOR, "HTTP GET")
    draw_node(draw, 570, 260, nw, nh, "Any Entries?", IF_COLOR, "IF condition")
    draw_node(draw, 810, 180, nw, nh, "Format Markdown Table", CODE_COLOR, "Code")
    draw_node(draw, 810, 370, nw, nh, "No Entries Today", NOOP_COLOR, "NoOp")
    draw_node(draw, 1000, 100, nw, nh, "Post Digest to Slack", SLACK_COLOR, "Slack")
    draw_node(draw, 1000, 260, nw, nh, "Email Digest", EMAIL_COLOR, "Email Send")

    draw_connection(draw, 250, 285, 310, 285)
    draw_connection(draw, 510, 285, 570, 285)
    draw_connection(draw, 770, 270, 810, 205, "true")
    draw_connection(draw, 770, 300, 810, 395, "false")
    draw_connection(draw, 1010, 205, 1000, 125)
    draw_connection(draw, 1010, 230, 1000, 285)

    draw.text((20, h - 30), "Placeholder — import daily-low-confidence-digest.json into n8n for live canvas", fill=(100, 100, 100))

    img.save(os.path.join(SCREENSHOTS_DIR, "n8n-daily-digest.png"))
    print("Created n8n-daily-digest.png")


if __name__ == "__main__":
    gen_doc_ingestion()
    gen_slack_ask()
    gen_daily_digest()
    print("All screenshots generated.")
