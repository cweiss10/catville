from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage
from datetime import datetime
import random
import json
import re
from typing import Any, Dict, List, Optional


def extract_json(text):
    if isinstance(text, AIMessage):
        text = text.content
    if not isinstance(text, str):
        text = str(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return {}
        return {}

def sanitize_schedule_output(raw: str):
    if not raw or not raw.strip():
        return "[]"

    # Remove ```json fences if any
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    # Replace 'TBD' in date/time with empty string
    cleaned = re.sub(r"'date':\s*'TBD'", '"date": ""', cleaned)
    cleaned = re.sub(r"'time':\s*'TBD'", '"time": ""', cleaned)

    # Convert single quotes to double quotes for valid JSON
    cleaned = cleaned.replace("'", '"')

    # Remove trailing commas
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*\]", "]", cleaned)

    # If it's still empty, return empty list
    if not cleaned.strip():
        return "[]"

    return cleaned


DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y")
TIME_FORMATS = ("%H:%M", "%I%p", "%I %p", "%I:%M%p", "%I:%M %p")


def _task_key(item: Dict[str, Any]) -> str:
    return (
        f"{item.get('date', '')}|{item.get('time', '')}|"
        f"{item.get('location', '')}|{item.get('commitment', '')}"
    )


def _parse_schedule_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip().upper()
    if not date_str or not time_str:
        return None

    date_obj = None
    for fmt in DATE_FORMATS:
        try:
            date_obj = datetime.strptime(date_str, fmt).date()
            break
        except ValueError:
            continue
    if date_obj is None:
        return None

    # Compact values like "5 PM" -> "5PM"
    time_str = re.sub(r"\s+", "", time_str)

    time_obj = None
    for fmt in TIME_FORMATS:
        try:
            time_obj = datetime.strptime(time_str, fmt).time()
            break
        except ValueError:
            continue
    if time_obj is None:
        return None

    return datetime.combine(date_obj, time_obj)

class Agent:
    def __init__(self, name, personality, world, llm, schedule):
        self.name = name
        self.personality = personality
        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)
        self.location = "home"
        self.world = world
        self.llm = llm
        self.schedule = self.normalize_schedule(schedule)
        self.completed_tasks: List[Dict[str, Any]] = []
        # optional: set externally by your factory
        self.relationships = getattr(self, "relationships", {})

    def observe(self):
        return f"{self.name} is at the {self.location}."

    def normalize_schedule(self, schedule: Any) -> List[Dict[str, Any]]:
        if isinstance(schedule, str):
            try:
                schedule = json.loads(schedule)
            except json.JSONDecodeError:
                try:
                    schedule = extract_json(schedule)
                except Exception:
                    schedule = []
        if isinstance(schedule, dict):
            schedule = [schedule]
        if not isinstance(schedule, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in schedule:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "date": (item.get("date") or "").strip(),
                    "time": (item.get("time") or "").strip(),
                    "location": (item.get("location") or "").strip(),
                    "commitment": (item.get("commitment") or "").strip(),
                    "status": item.get("status", "pending"),
                    "completed_at": item.get("completed_at", ""),
                }
            )
        return normalized

    def get_due_task(self) -> Optional[Dict[str, Any]]:
        now_label = self.world.get("time", "")
        try:
            now = datetime.strptime(now_label, "%Y-%m-%d %H:%M")
        except ValueError:
            return None

        for item in self.schedule:
            if item.get("status", "pending") == "completed":
                continue
            task_dt = _parse_schedule_datetime(item.get("date", ""), item.get("time", ""))
            if task_dt and task_dt == now:
                return item
        return None

    def complete_task(self, task: Dict[str, Any]) -> None:
        task["status"] = "completed"
        task["completed_at"] = self.world.get("time", "")
        self.completed_tasks.append(
            {
                "date": task.get("date", ""),
                "time": task.get("time", ""),
                "location": task.get("location", ""),
                "commitment": task.get("commitment", ""),
                "completed_at": task.get("completed_at", ""),
            }
        )

    def format_upcoming_schedule(self, limit: int = 3) -> str:
        pending = [s for s in self.schedule if s.get("status", "pending") != "completed"]
        if not pending:
            return "none"
        lines = [
            f"{s.get('date', '')} {s.get('time', '')} @ {s.get('location', '')}: {s.get('commitment', '')}"
            for s in pending[:limit]
        ]
        return " | ".join(lines)

    def format_recent_completions(self, limit: int = 3) -> str:
        if not self.completed_tasks:
            return "none"
        lines = [
            f"{t.get('completed_at', '')} @ {t.get('location', '')}: {t.get('commitment', '')}"
            for t in self.completed_tasks[-limit:]
        ]
        return " | ".join(lines)

    def decide_action(self):
        due_task = self.get_due_task()
        if due_task:
            dest = due_task.get("location", "").strip()
            if dest in self.world["locations"]:
                return f"go to {dest}", due_task
            return "stay", due_task

        locations = list(self.world["locations"].keys())
        nav_actions = [f"go to {loc}" for loc in locations]
        actions = ["stay"] + nav_actions
        return random.choice(actions), None

    def move(self, new_location):
        # Guard against missing names (first boot) or mismatched lists
        if self.name in self.world["locations"].get(self.location, []):
            self.world["locations"][self.location].remove(self.name)
        self.location = new_location
        self.world["locations"][new_location].append(self.name)

    
    def build_schedule(self, conversation, schedule, time_label):
        """Using the agent's memories, build a schedule for commitments that they need"""
        prompt = PromptTemplate(
            input_variables= [
                "time_label", "conversation", "schedule", "completed"
            ],
            template=(
                "The current date and time is: {time_label}"
                "Given the content of this conversation: {conversation} " 
                "And this existing schedule: {schedule} "
                "And these already completed commitments: {completed} "
                "Add any additional commitments from this conversation to the schedule in the same JSON format. An example schedule should look like this: "
                "{{"
                "  'date': '10/12/2024',"
                "  'time': '5PM',"
                "  'location': 'cafe',"
                "  'commitment': 'attend an art show with Juan',"
                "}}, "
                "{{"
                "  'date': '10/25/2024',"
                "  'time': '8PM',"
                "  'location': 'library',"
                "  'commitment': 'attend halloween party with Marge',"
                "}}"
                "The locations can only be from this list: [park, cafe, library, school, hospital, market, town_hall, theater, gym, museum, restaurant, train_station]"
                "Keep existing schedule items. Do not include any commitment that is already completed."
                "Return the full schedule. RETURN JSON ONLY AND NO OTHER MESSAGE."
            ),
        )
        chain = prompt | self.llm
        result = chain.invoke({
            "time_label": time_label,
            "conversation": conversation,
            "schedule": schedule,
            "completed": self.format_recent_completions(limit=10),
        })
        raw_output = getattr(result, "content", None) or str(result)

        # Sanitize
        cleaned = sanitize_schedule_output(raw_output)

        # Parse safely
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # If still fails, just fallback to empty list
            parsed = []

        if isinstance(parsed, dict) and "schedule" in parsed:
            parsed = parsed["schedule"]
        normalized_new = self.normalize_schedule(parsed)
        if not normalized_new:
            return

        existing = self.normalize_schedule(self.schedule)
        existing_by_key = {_task_key(item): item for item in existing}
        merged: List[Dict[str, Any]] = []
        merged_keys = set()
        for item in normalized_new:
            if item.get("date") in ("TBD", "", None):
                item["date"] = datetime.today().strftime("%Y-%m-%d")
            if item.get("time") in ("TBD", "", None):
                item["time"] = "00:00"  # or some default hour
            key = _task_key(item)
            if key in existing_by_key and existing_by_key[key].get("status") == "completed":
                item["status"] = "completed"
                item["completed_at"] = existing_by_key[key].get("completed_at", "")
            merged.append(item)
            merged_keys.add(key)

        for item in existing:
            if item.get("status") == "completed":
                key = _task_key(item)
                if key not in merged_keys:
                    merged.append(item)

        self.schedule = merged


    def interact(self, other_agent, commitment):
        """Create a short conversation that leverages relationship hints,
        current location/time, and each agent's recent memory summary.
        """
        # Context from world
        location = self.location
        time_label = self.world.get("time", "")

        # Relationship hints (seeded by your factory)
        hint1 = (getattr(self, "relationships", {}) or {}).get(other_agent.name, "")
        hint2 = (getattr(other_agent, "relationships", {}) or {}).get(self.name, "")

        # Recent summaries (helps continuity between hourly runs)
        sum1 = getattr(self.memory, "moving_summary_buffer", "") or ""
        sum2 = getattr(other_agent.memory, "moving_summary_buffer", "") or ""

        # Compose a compact context string
        context_lines = []
        if hint1 or hint2:
            context_lines.append(
                f"Relationship hints — {self.name}→{other_agent.name}: {hint1} "
                f"| {other_agent.name}→{self.name}: {hint2}"
            )
        if sum1 or sum2:
            context_lines.append(
                f"Recent memories — {self.name}: {sum1} | {other_agent.name}: {sum2}"
            )
        context = " ".join(context_lines).strip()

        # Prompt with structure + guardrails
        prompt = PromptTemplate(
            input_variables=[
                "name1", "name2", "personality1", "personality2",
                "location", "time_label", "context", "commitment",
                "schedule1", "schedule2", "completed1", "completed2"
            ],
            template=(
                "{name1} and {name2} meet at the {location} around {time_label}. "
                "{name1} is {personality1}. {name2} is {personality2}. "
                "{context}\n\n"
                "{name1}'s upcoming schedule: {schedule1}. "
                "{name2}'s upcoming schedule: {schedule2}. "
                "{name1}'s recently completed commitments: {completed1}. "
                "{name2}'s recently completed commitments: {completed2}. "
                "They are here for this commitment right now: {commitment}. "
                "Write a short, natural back-and-forth conversation (4–8 lines). "
                "Make it grounded and specific to the setting and their relationship history when relevant. If a plan is made, decide on a time and place to meet to add to their schedules. The meeting time can ONLY be on the hour exactly."
                "Format strictly as 'Name: utterance' per line."
            ),
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "name1": self.name,
            "name2": other_agent.name,
            "personality1": self.personality,
            "personality2": other_agent.personality,
            "location": location,
            "time_label": time_label,
            "context": context,
            "commitment": commitment,
            "schedule1": self.format_upcoming_schedule(),
            "schedule2": other_agent.format_upcoming_schedule(),
            "completed1": self.format_recent_completions(),
            "completed2": other_agent.format_recent_completions(),
        })

        conversation = getattr(result, "content", None) or str(result)
        self.build_schedule( conversation, self.schedule, time_label)
        other_agent.build_schedule( conversation, other_agent.schedule, time_label)
        # Save a concise input/output pair so ConversationSummaryBufferMemory
        # can keep a rolling context across hourly runs
        self.memory.save_context(
            {"input": f"Talked to {other_agent.name} at {location} around {time_label}"},
            {"output": conversation}
        )
        other_agent.memory.save_context(
            {"input": f"Talked to {self.name} at {location} around {time_label}"},
            {"output": conversation}
        )

        print(conversation)
        return conversation
