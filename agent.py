from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
import random


# === Agent Class ===
class Agent:
    def __init__(self, name, personality, world, llm):
        self.name = name
        self.personality = personality
        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
        self.location = "home"
        self.world = world
        self.llm = llm

    def observe(self):
        return f"{self.name} is at the {self.location}."

    def decide_action(self):
        actions = ["stay", "go to park", "go to cafe"]
        return random.choice(actions)

    def move(self, new_location):
        self.world["locations"][self.location].remove(self.name)
        self.location = new_location
        self.world["locations"][new_location].append(self.name)

    def reflect(self):
        summary = self.memory.buffer[-1]["content"] if self.memory.buffer else "Nothing to reflect."
        return f"{self.name} reflects: {summary}"

    def interact(self, other_agent):
        prompt = PromptTemplate(
            input_variables=["name1", "name2", "personality1", "personality2"],
            template=(
                "{name1} and {name2} meet at a location. "
                "{name1} is {personality1}. {name2} is {personality2}. "
                "Write a short, natural conversation between them."
            )
        )
        chain = prompt | self.llm
        result = chain.invoke({
            "name1": self.name,
            "name2": other_agent.name,
            "personality1": self.personality,
            "personality2": other_agent.personality
        })

        conversation = result.content if hasattr(result, "content") else str(result)
        self.memory.save_context({"input": f"Talked to {other_agent.name}"}, {"output": conversation})
        other_agent.memory.save_context({"input": f"Talked to {self.name}"}, {"output": conversation})
        print(conversation)