from typing import List, Callable
import anthropic
import os
import json
from datetime import datetime

class Agent():
    def __init__(
        self,
        system_prompt: str,
        tools: dict[str, Callable],
        provider_tools_list: List[str],
        user_query: str
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.provider_tools_list = provider_tools_list
        self.messages = []

        self.llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        self.messages.append({"role": "user", "content": [{"type": "text", "text": user_query}]})
    
    def _check_if_done(self):
        last_message = self.messages[-1]
        for block in last_message["content"]:
            if block["type"] == "tool_use" and block["name"] == "output":
                return True
        return False
    
    async def _handle_tool(self, tool_name: str, tool_input: str):
        tool = self.tools[tool_name]
        tool_output = tool(tool_input)
        return tool_output
    
    async def generate_response(self):
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "system": self.system_prompt,
            "messages": self.messages,
            "max_tokens": 2048,
            "tools": self.provider_tools_list,
            "tool_choice": {"type": "any"}
        }

        response = self.llm.messages.create(**data)

        content = []
        for block in response.content:
            print(block)
            if block.type == "text":
                content.append(
                    {
                        "type": "text",
                        "text": block.text
                    }
                )
            elif block.type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "name": block.name,
                        "id": block.id,
                        "input": block.input
                    }
                )
        return content
    
    async def run(self):
        done = False
        while not done:
            response = await self.generate_response()
            self.messages.append({"role": "assistant", "content": response})
            for block in response:
                if block["type"] == "tool_use":
                    if block["name"] == "output":
                        print(block["input"]["content"])
                        done = True
                        break

                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_output = await self._handle_tool(tool_name, tool_input)
                    self.messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": block["id"], "content": tool_output}
                    ]})

        # Create a 'logs' directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # Generate a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/conversation_{timestamp}.jsonl"

        # Append all messages to the JSONL file
        with open(filename, 'a') as f:
            for message in self.messages:
                json.dump(message, f)
                f.write('\n')
                    