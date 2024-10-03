import asyncio
import json
from Agent import Agent
from tools.perplexity import Perplexity
from tools.scratchpad import Scratchpad
from tools.output import Output
from tools.python_sandbox import PythonSandbox

perplexity = Perplexity()
scratchpad = Scratchpad()
output = Output()
python_sandbox = PythonSandbox()

tools = {
    "perplexity": perplexity.run,
    "write_scratchpad": scratchpad.write,
    "read_scratchpad": scratchpad.read,
    "output": output.run,
    "python": python_sandbox.run
}

provider_tools_list = [
    {
        "name": "perplexity",
        "description": "Sends a query to the Perplexity AI model and returns its response. This tool is useful for answering general questions, providing information, or generating content based on the given query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or prompt to send to the Perplexity AI model."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "write_scratchpad",
        "description": "Writes content to a scratchpad, appending it to any existing content. This tool is useful for storing temporary information or building up a response over multiple steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to write to the scratchpad."
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "read_scratchpad",
        "description": "Reads and returns the current content of the scratchpad. This tool is useful for retrieving previously stored information or checking the current state of the scratchpad.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "python",
        "description": "Executes Python code in a sandboxed environment. This tool is useful for running code that requires file system access, network access, or other system resources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "output",
        "description": "Output completes the task and provides one final response to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The final response to the user."
                }
            },
            "required": ["content"]
        }
    },
]

async def run_agent(prompt):
    agent = Agent(
        system_prompt="You are a helpful assistant",
        tools=tools,
        provider_tools_list=provider_tools_list,
        user_query=prompt
    )
    return await agent.run()

async def main():
    CHECKPOINT_COUNT = 10
    
    prompts = [
        "Design a circular economy framework for the recycling and repurposing of decommissioned wind turbine blades.",
        "Propose innovative strategies for reducing electronic waste in the smartphone industry.",
        "Develop a comprehensive plan for sustainable urban transportation in a rapidly growing city."
    ]


    all_runs = []
    for prompt in prompts:
        if len(all_runs) % CHECKPOINT_COUNT == 0:
            print(f"Checkpoint {len(all_runs)}")
            with open('agent_runs.json', 'w') as f:
                json.dump(all_runs, f, indent=2)
                
        messages = await run_agent(prompt)
        all_runs.append(messages)

    with open('agent_runs.json', 'w') as f:
        json.dump(all_runs, f, indent=2)

asyncio.run(main())