# from agent import Agent
from catville import tick
from catville import DEFAULT_WORLD, default_agents_factory
from langchain_ollama import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory
from state_io import load_state

llm = ChatOllama(model="mistral")

world, agents = load_state(llm, DEFAULT_WORLD, default_agents_factory)
tick()

for agent in agents:
    print(f"NAME: {agent.name}")
    print(f"schedule: {agent.schedule}")