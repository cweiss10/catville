# Catville
In 2023, Stanford and Google introduced a simulation of human behavior using AI agents. You can read the original paper [here](https://arxiv.org/abs/2304.03442).

# What Is Catville?
Inspired by that research, Catville is my own take on an autonomous, AI-driven social simulation. The project uses Mistral's open-source LLM for its strong performance and accessibility (free to run locally).

The simulation evolves on its own: each day, the agents interact, make decisions, and reflect—forming an emergent narrative of society.

# Daily Updates via Newsletter
Every day, a new Buttondown post is generated, summarizing:

- Key interactions between the people of Catville

- A narrative overview of Catville’s daily events

[Click here to subscribe to the newsletter and receive daily updates!](https://buttondown.com/catherineweiss95)


### Running Locally

1. **Start Ollama with Mistral**

   Make sure Ollama is installed and the Mistral model is pulled:

   ```bash
   ollama pull mistral
   ```

   Start Ollama daemon (if not already running):

    ```bash
    ollama run mistral
    ```


    Install all the required dependencies using Poetry:

    ```bash
    poetry install
    ```

    Run the Simulation:

    ```bash
    poetry run python catville.py
    ```