from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
import random


class Agent:
    def __init__(self, name, personality, world, llm):
        self.name = name
        self.personality = personality
        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)
        self.location = "home"
        self.world = world
        self.llm = llm
        # optional: set externally by your factory
        self.relationships = getattr(self, "relationships", {})

    def observe(self):
        return f"{self.name} is at the {self.location}."

    def decide_action(self):
        # Build actions dynamically from world locations
        locations = list(self.world["locations"].keys())
        nav_actions = [f"go to {loc}" for loc in locations]
        actions = ["stay"] + nav_actions
        return random.choice(actions)

    def move(self, new_location):
        # Guard against missing names (first boot) or mismatched lists
        if self.name in self.world["locations"].get(self.location, []):
            self.world["locations"][self.location].remove(self.name)
        self.location = new_location
        self.world["locations"][new_location].append(self.name)

    def reflect(self):
        summary = getattr(self.memory, "moving_summary_buffer", "") or "Nothing to reflect."
        return f"{self.name} reflects: {summary}"
    
    def build_schedule(conversation, schedule, agent):
        """Using the agent's memories, build a schedule for commitments that they need"""
        template = f"""Given the content of this conversation: {conversation}
        And this existing schedule: {schedule}
        Add any additional commitments from this conversation to the schedule in the same JSON format. Return the full schedule.
        """

    def interact(self, other_agent):
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
                "location", "time_label", "context"
            ],
            template=(
                "{name1} and {name2} meet at the {location} around {time_label}. "
                "{name1} is {personality1}. {name2} is {personality2}. "
                "{context}\n\n"
                "Write a short, natural back-and-forth conversation (4–8 lines). "
                "Make it grounded and specific to the setting and their relationship history when relevant. "
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
        })

        conversation = getattr(result, "content", None) or str(result)

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
