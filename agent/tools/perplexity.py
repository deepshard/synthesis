import os
import requests
from dotenv import load_dotenv

load_dotenv()

class Perplexity:
    def __init__(self):
        self.system_prompt = ""
        self.model = "llama-3.1-sonar-large-128k-online"
        self.url = "https://api.perplexity.ai/chat/completions"

    def run(self, args):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": args["query"]}
        ]

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("PERPLEXITY_API_KEY")
        }
        response = requests.post(self.url, headers=headers, json={"model": self.model, "messages": messages})
        return response.json()["choices"][0]["message"]["content"]
        
        