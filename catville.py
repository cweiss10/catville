from agent import Agent
from langchain_ollama import ChatOllama
from datetime import datetime, timedelta
from state_io import load_state, save_state


# === Default World (used on first run or if state missing) ===
DEFAULT_WORLD = {
    "locations": {
        "home": [],
        "park": [],
        "cafe": [],
        "library": [],
        "school": [],
        "hospital": [],
        "market": [],
        "town_hall": [],
        "theater": [],
        "gym": [],
        "museum": [],
        "restaurant": [],
        "train_station": [],
    },
    "time": "08:00 AM",
}


def default_agents_factory(world, llm):
    agents = [
        Agent(
            "Andy",
            "a cheerful, early-rising artist who sketches the sunrise at the park, paints watercolor murals, "
            "and mentors kids at the community center; carries a dented thermos of tea, fears creative block, "
            "and dreams of a solo gallery show",
            world, llm
        ),
        Agent(
            "Samantha",
            "a quiet, methodical programmer who prefers solitude and noise-cancelling headphones; maintains open-source "
            "libraries at night, leaves kind notes instead of small talk, and feeds the neighborhood strays behind the cafe; "
            "finds comfort in routines and logic puzzles",
            world, llm
        ),
        Agent(
            "Caroline",
            "a purple-haired barista and latte-art whisperer who remembers everyone’s order; a compassionate gossip filter "
            "who connects people gently; moonlights as an indie singer at the theater and guards the cafe’s cozy vibe fiercely",
            world, llm
        ),
        Agent(
            "Peter",
            "a gym-obsessed ex-finance hopeful with a fragile ego and a bookshelf of self-help; talks a big game about startups, "
            "is secretly soft with kids at the park, and insists he’s looking for real love even while networking at every table",
            world, llm
        ),
        Agent(
            "Mei",
            "a meticulous librarian and amateur archivist who labels everything, champions quiet spaces, and runs a Saturday "
            "zine club; keeps seeds in her tote for the community garden and believes small rituals make strong towns",
            world, llm
        ),
        Agent(
            "Diego",
            "a disciplined fitness coach and part-time EMT who times his runs to the minute; pragmatic in emergencies, warm with "
            "beginners at the gym, and still rehabbing an old knee injury; swears by stretching and street tacos",
            world, llm
        ),
        Agent(
            "Noor",
            "an energetic science teacher who runs the school’s robotics club; turns everyday moments into experiments, "
            "keeps snacks for overwhelmed students, and advocates at town hall for better lab equipment",
            world, llm
        ),
        Agent(
            "Leo",
            "a wandering violinist and theater tech who busks near the market; improvises duets with birds at the park, "
            "collects stories for a future musical, and fixes broken amps with improbable spare parts",
            world, llm
        ),
    ]

    # Add relationship hints (could also load from a JSON file later)
    relationships = {
        "Andy": {
            "Caroline": "Andy often sketches Caroline while she works at the cafe; they share art talk.",
            "Mei": "Andy visits Mei at the library for inspiration and research on local artists."
        },
        "Samantha": {
            "Caroline": "Caroline always makes Samantha’s late-night coffee and teases her about her headphones.",
            "Leo": "Samantha has debugged Leo’s amp wiring once; he owes her a song."
        },
        "Caroline": {
            "Peter": "Caroline once rejected Peter’s attempt to flirt at the cafe; awkward tension lingers.",
            "Leo": "Caroline and Leo sometimes perform together at the theater."
        },
        "Peter": {
            "Diego": "Peter trains with Diego at the gym but exaggerates his lifts.",
            "Noor": "Peter admires Noor’s confidence at town hall, though she finds him exhausting."
        },
        "Mei": {
            "Noor": "Mei helps Noor’s students find science books; they respect each other deeply.",
        },
        "Diego": {
            "Leo": "Diego spotted Leo first aid once after a street performance injury.",
        },
        "Noor": {
            "Samantha": "Noor is curious about Samantha’s coding; they plan a robotics–software project.",
        },
        "Leo": {
            "Andy": "Leo and Andy trade sketches and songs in the park.",
        },
    }

    for agent in agents:
        agent.relationships = relationships.get(agent.name, {})

    return agents

# === Boot ===
llm = ChatOllama(model="mistral")
world, agents = load_state(llm, DEFAULT_WORLD, default_agents_factory)

# Ensure initial occupancy if fresh
for agent in agents:
    if agent.name not in world["locations"][agent.location]:
        world["locations"][agent.location].append(agent.name)


def parse_time_label(t: str) -> datetime:
    # "08:00 AM" → today’s date + that time (date is irrelevant for sim)
    return datetime.strptime(t, "%I:%M %p")


def format_time_label(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


# === Simulation Tick ===
def tick():
    print(f"\n--- {world['time']} ---")

    # 1) Each agent decides and (maybe) moves
    for agent in agents:
        action = agent.decide_action()
        if action == "stay":
            continue

        if action.startswith("go to "):
            dest = action[len("go to ") :].strip()
            # ensure destination exists
            if dest in world["locations"]:
                agent.move(dest)

        print(agent.observe())

    # 2) Interactions (pairwise per location, simple pairing)
    for loc in world["locations"]:
        present = [a for a in agents if a.location == loc]
        if len(present) >= 2:
            present[0].interact(present[1])

    # 3) Advance time by one hour, rolling AM/PM properly
    try:
        t = parse_time_label(world["time"])
    except ValueError:
        # Fallback if someone edits state manually
        t = datetime.strptime("08:00 AM", "%I:%M %p")
    t = t + timedelta(hours=1)
    world["time"] = format_time_label(t)

    # 4) Persist state (world + agents + their memories)
    save_state(world, agents)


# Run a single tick
if __name__ == "__main__":
    tick()