import json
from pathlib import Path
from typing import Dict, List, Tuple
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

STATE_PATH = Path("state/state.json")


def ensure_state_dir():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def serialize_world(world: Dict) -> Dict:
    # world is already JSON-safe as you’ve defined it
    return {
        "locations": world["locations"],
        "time": world["time"],
    }


def serialize_agents(agents: List) -> List[Dict]:
    out = []
    for a in agents:
        mem = a.memory
        # Keep it compact: last ~20 messages + rolling summary
        messages = messages_to_dict(getattr(mem.chat_memory, "messages", []))[-20:]
        out.append(
            {
                "name": a.name,
                "personality": a.personality,
                "location": a.location,
                "memory": {
                    "summary": getattr(mem, "moving_summary_buffer", "") or "",
                    "messages": messages,
                },
            }
        )
    return out


def save_state(world: Dict, agents: List) -> None:
    ensure_state_dir()
    data = {
        "world": serialize_world(world),
        "agents": serialize_agents(agents),
    }
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_state(llm, default_world: Dict, default_agents_factory) -> Tuple[Dict, List]:
    """Load state if present; else return defaults.
    default_agents_factory(): -> List[Agent] (creates fresh agents with the given llm/world)
    """
    if not STATE_PATH.exists():
        return default_world, default_agents_factory(default_world, llm)

    with STATE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Rebuild world
    world = data.get("world", default_world)
    # Ensure all defined locations exist (in case you add more later)
    for loc in default_world["locations"].keys():
        world["locations"].setdefault(loc, [])

    # Rebuild agents
    saved_agents = data.get("agents", [])
    agents = []
    for sa in saved_agents:
        from agent import Agent  # local import to avoid circular during type checking
        a = Agent(sa["name"], sa["personality"], world, llm)
        a.location = sa.get("location", "home")

        # Rehydrate memory
        mem_blob = sa.get("memory", {})
        mem = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
        mem.moving_summary_buffer = mem_blob.get("summary", "") or ""
        mem.chat_memory.messages = messages_from_dict(mem_blob.get("messages", []))
        a.memory = mem

        agents.append(a)

    # Rebuild world occupancy from agent locations (source of truth)
    for loc in world["locations"]:
        world["locations"][loc] = []
    for a in agents:
        world["locations"][a.location].append(a.name)

    return world, agents