from agent import agent
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="mistral")
# === Simulation Setup ===
agents = [
    Agent("Andy", "a cheerful artist who loves mornings", llm),
    Agent("Samantha", "a quiet programmer who enjoys solitude", llm),
    Agent("Caroline", "a purple haired barista who makes the best lattes", llm),
    Agent("Peter", "a narcissistic finance bro who didn't make it to Wall Street but wants to find true love", llm),
]

# Place agents in initial location
for agent in agents:
    world["locations"][agent.location].append(agent.name)

# === Simulation Loop ===
def tick():
    print(f"\n--- {world['time']} ---")
    for agent in agents:
        print(agent.observe())
        action = agent.decide_action()
        if action == "stay":
            continue
        elif action == "go to park":
            agent.move("park")
        elif action == "go to cafe":
            agent.move("cafe")

    # Interactions
    for loc in world["locations"]:
        present = [a for a in agents if a.location == loc]
        if len(present) >= 2:
            present[0].interact(present[1])

    # Advance time (mocked)
    current_hour = int(world["time"].split(":"[0]))
    world["time"] = f"{current_hour + 1}:00 AM"

# Run a few ticks
for _ in range(3):
    tick()
    time.sleep(1)  # simulate real time (can be removed)