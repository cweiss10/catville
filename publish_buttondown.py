# publish_buttondown.py
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os, sys, json, requests

LA = ZoneInfo("America/Los_Angeles")
API_URL = "https://api.buttondown.com/v1/emails"
API_KEY = os.environ.get("BUTTONDOWN_API_KEY")

def path_for(date):
    mm = date.strftime("%m"); dd = date.strftime("%d"); yyyy = date.strftime("%Y")
    return Path(f"summaries/{mm}/{dd}/{yyyy}.md"), f"Catville Daily — {yyyy}-{mm}-{dd}"

def main():
    if not API_KEY:
        print("Missing BUTTONDOWN_API_KEY", file=sys.stderr)
        sys.exit(1)

    today_la = datetime.now(tz=LA).date()
    date = today_la - timedelta(days=1)
    md_path, subject = path_for(date)
    if not md_path.exists():
        print(f"No summary for {date} at {md_path}")
        return

    body_md = md_path.read_text(encoding="utf-8")

    # Publish ~2 minutes from now (Buttondown recommends scheduling with publish_date)
    publish_dt = datetime.now(tz=LA) + timedelta(minutes=2)
    # Buttondown expects ISO8601 / UTC; convert to UTC
    publish_utc = publish_dt.astimezone(ZoneInfo("UTC")).isoformat()

    payload = {
        "subject": subject,
        "body": body_md,               # Markdown is supported
        "status": "scheduled",         # queue it
        "publish_date": publish_utc,   # when to send
        "email_type": "public",        # show in web archive
    }

    headers = {"Authorization": f"Token {API_KEY}"}
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code not in (200, 201):
        print("Buttondown API error:", resp.status_code, resp.text, file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    print("Scheduled:", data.get("subject"), "→", data.get("absolute_url", "(no url)"))

if __name__ == "__main__":
    main()
