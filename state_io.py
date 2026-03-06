# state_io.py
import json
import ast
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import messages_from_dict, BaseMessage

STATE_PATH = Path("state/state.json")


def ensure_state_dir():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _message_to_serializable(m):
    """
    Produce a dict compatible with langchain.schema.messages_to_dict
    so messages_from_dict can rehydrate later.
    """
    if isinstance(m, BaseMessage):
        cls = m.__class__.__name__.lower()
        if "human" in cls:
            t = "human"
        elif "ai" in cls:
            t = "ai"
        elif "system" in cls:
            t = "system"
        else:
            t = cls
        data = {"content": getattr(m, "content", "")}
        # include additional kwargs if present
        additional = getattr(m, "additional_kwargs", None)
        if additional:
            data["additional_kwargs"] = additional
        return {"type": t, "data": data}
    if isinstance(m, dict):
        return m
    # fallback to string
    return {"type": "unknown", "data": {"content": str(m)}}


def serialize_agents(agents: List) -> List[Dict]:
    serialized = []
    for a in agents:
        mem = getattr(a, "memory", None)
        # messages
        messages = []
        if mem and hasattr(mem, "chat_memory"):
            raw_messages = getattr(mem.chat_memory, "messages", []) or []
            # defensive conversion
            messages = [_message_to_serializable(m) for m in raw_messages][-20:]

        # summary may be a BaseMessage in some langchain versions
        summary_val = getattr(mem, "moving_summary_buffer", "") if mem else ""
        if isinstance(summary_val, BaseMessage):
            summary_val = getattr(summary_val, "content", str(summary_val))
        elif not isinstance(summary_val, str):
            summary_val = str(summary_val)

        # normalize schedule into a JSON-safe object if possible
        sched = getattr(a, "schedule", {})
        normalized_schedule = sched
        if isinstance(sched, str):
            try:
                normalized_schedule = json.loads(sched)
            except Exception:
                try:
                    normalized_schedule = ast.literal_eval(sched)
                except Exception:
                    normalized_schedule = sched
        elif not isinstance(sched, (dict, list, str, int, float, bool, type(None))):
            normalized_schedule = str(sched)

        serialized.append(
            {
                "name": a.name,
                "personality": getattr(a, "personality", ""),
                "location": getattr(a, "location", ""),
                "schedule": normalized_schedule,
                "completed_tasks": getattr(a, "completed_tasks", []),
                "memory": {"summary": summary_val or "", "messages": messages},
                "relationships": getattr(a, "relationships", {}),
            }
        )
    return serialized


def normalize_time(world):
    time_str = world.get("time", "08:00 AM")
    try:
        # Try full date + time
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        # Only time exists → assume today or a default date
        dt = datetime.strptime(datetime.today().strftime("%Y-%m-%d ") + time_str, "%Y-%m-%d %H:%M")
    world["time"] = dt.strftime("%Y-%m-%d %H:%M")


def serialize_world(world: Dict) -> Dict:
    return {"locations": world["locations"], "time": world["time"]}


def save_state(world: Dict, agents: List) -> None:
    ensure_state_dir()
    data = {"world": serialize_world(world), "agents": serialize_agents(agents)}

    # atomic write to avoid truncated/corrupt json
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(str(tmp), str(STATE_PATH))


def load_state(llm, default_world: Dict, default_agents_factory) -> Tuple[Dict, List]:
    """Load state if present; else return defaults.
    default_agents_factory(world, llm) -> List[Agent]
    """
    if not STATE_PATH.exists():
        return default_world, default_agents_factory(default_world, llm)

    with STATE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # rebuild world
    world = data.get("world", default_world)
    for loc in default_world["locations"].keys():
        world["locations"].setdefault(loc, [])

    # rebuild agents
    saved_agents = data.get("agents", [])
    agents = []
    for sa in saved_agents:
        from agent import Agent  # local import

        a = Agent(sa["name"], sa.get("personality", ""), world, llm, sa.get("schedule", {}))
        a.location = sa.get("location", "home")
        a.completed_tasks = sa.get("completed_tasks", []) or []

        # rehydrate memory
        mem_blob = sa.get("memory", {})
        mem = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
        mem.moving_summary_buffer = mem_blob.get("summary", "") or ""
        # messages_from_dict expects the list/dict format we wrote above
        msgs = mem_blob.get("messages", []) or []
        try:
            mem.chat_memory.messages = messages_from_dict(msgs)
        except Exception:
            # as a fallback, store minimal string messages
            mem.chat_memory.messages = []
            for m in msgs:
                try:
                    c = m.get("data", {}).get("content", str(m))
                except Exception:
                    c = str(m)
                # create a minimal rehydrated message dict that messages_from_dict can't parse
                mem.chat_memory.messages.append(c)
        a.memory = mem

        agents.append(a)

    # rebuild occupancy
    for loc in world["locations"]:
        world["locations"][loc] = []
    for a in agents:
        world["locations"].setdefault(a.location, [])
        world["locations"][a.location].append(a.name)

    return world, agents
