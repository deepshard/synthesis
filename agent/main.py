import asyncio
from Agent import Agent
from tools.perplexity import Perplexity
from tools.scratchpad import Scratchpad
from tools.output import Output

perplexity = Perplexity()
scratchpad = Scratchpad()
output = Output()
tools = {
    "perplexity": perplexity.run,
    "write_scratchpad": scratchpad.write,
    "read_scratchpad": scratchpad.read,
    "output": output.run
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
    }
]

agent = Agent(
    system_prompt="You are a helpful assistant",
    tools=tools,
    provider_tools_list=provider_tools_list,
    user_query="What is the capital of the moon?"
)
asyncio.run(agent.run())