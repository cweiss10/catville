# daily_summary.py
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json

from langchain_ollama import ChatOllama

LA_TZ = ZoneInfo("America/Los_Angeles")

def path_for(date):
    mm = date.strftime("%m")
    dd = date.strftime("%d")
    yyyy = date.strftime("%Y")
    log_path = Path(f"logs/{mm}/{dd}/{yyyy}.txt")
    summary_path = Path(f"summaries/{mm}/{dd}/{yyyy}.md")
    return log_path, summary_path

def read_agent_summaries(state_path=Path("state/state.json")):
    """Pull rolling memory summaries per agent (if available)."""
    if not state_path.exists():
        return []
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for a in data.get("agents", []):
        out.append({
            "name": a.get("name", "Unknown"),
            "personality": a.get("personality", ""),
            "location": a.get("location", ""),
            "summary": (a.get("memory", {}) or {}).get("summary", "")
        })
    return out

def update_index():
    """Rebuild summaries/index.md with links to all available daily summaries."""
    base = Path("summaries")
    index_path = base / "index.md"
    if not base.exists():
        return

    entries = []
    for path in sorted(base.rglob("*.md")):
        if path.name == "index.md":
            continue
        # derive date string from path parts: summaries/MM/DD/YYYY.md
        try:
            mm, dd, filename = path.parts[-3], path.parts[-2], path.name
            yyyy = filename.replace(".md", "")
            date_str = f"{yyyy}-{mm}-{dd}"
            rel_path = path.relative_to(base)
            entries.append((date_str, rel_path))
        except Exception:
            continue

    # newest first
    entries.sort(reverse=True)

    lines = ["# Town Chronicles", ""]
    for date_str, rel_path in entries:
        lines.append(f"- [{date_str}]({rel_path.as_posix()})")

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    # Summarize the **previous calendar day** in LA.
    now_la = datetime.now(tz=LA_TZ)
    target_date = (now_la - timedelta(days=1)).date()
    log_path, summary_path = path_for(target_date)

    if not log_path.exists():
        # Nothing to do if the daily log wasn't created (e.g., first run).
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        # Idempotent: don't regenerate if it already exists.
        return

    log_text = log_path.read_text(encoding="utf-8")
    agent_summaries = read_agent_summaries()

    # Build a compact context for the LLM
    agents_block = "\n".join(
        f"- {a['name']} @ {a['location']}: {a['summary']}"
        for a in agent_summaries if a.get("summary")
    )

    prompt = f"""
You are the town chronicler. Summarize the day's events from the simulation logs below.

Constraints:
- Be concise but vivid (300–600 words).
- Prefer specifics (who, where, when) over generalities.
- Organize into sections: "Timeline Highlights", "Notable Conversations", "Agent Arcs", "Locations Activity", "Seeds for Tomorrow".
- Do NOT invent characters; only use names present.
- If information is missing, acknowledge briefly.

DATE: {target_date.strftime('%Y-%m-%d')}
AGENT MEMORY SNAPSHOTS:
{agents_block or '(no memory snapshots available)'}

RAW LOG (verbatim):
{log_text}
"""

    llm = ChatOllama(model="mistral")
    result = llm.invoke(prompt)
    content = getattr(result, "content", None) or str(result)

    summary_path.write_text(content.strip() + "\n", encoding="utf-8")
    if not summary_path.exists():
        log_text = log_path.read_text(encoding="utf-8")
        agent_summaries = read_agent_summaries()
        # build prompt etc...
        llm = ChatOllama(model="mistral")
        result = llm.invoke(prompt)
        content = getattr(result, "content", None) or str(result)
        summary_path.write_text(content.strip() + "\n", encoding="utf-8")
    
    update_index()

if __name__ == "__main__":
    main()
