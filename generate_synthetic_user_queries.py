import os
import requests
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import random
import re
import logging
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

in_depth_prompt = """
Assume the role of a Prompt Evolution Specialist. Your task is to comprehensively transform and elevate the provided prompt, enhancing its complexity, nuance, and depth. 
The goal is to create a version that significantly diverges from the original in structure and phrasing while maintaining or expanding its core intent. This evolved prompt 
should be designed to challenge and stimulate advanced language models (such as GPT-4 and its contemporaries) in novel ways, encouraging more nuanced, creative, and in-depth 
responses. Consider incorporating multifaceted objectives, intricate scenarios, or thought-provoking philosophical angles. The rewritten prompt should be substantially 
different from the original to minimize similarity when subjected to advanced textual analysis techniques like ROUGE scoring or cosine similarity comparisons.
"""

in_breadth_prompt = """
Embody the role of a Prompt Innovation Specialist. Your mission is to synthesize a novel prompt that draws subtle inspiration from the source prompt while significantly 
diverging in content and approach. The new creation should occupy a related conceptual space but explore a more uncommon or niche aspect within that domain. Maintain parity 
with the source prompt in terms of structural complexity and cognitive demand, ensuring the output is neither more simplistic nor overly abstruse. The innovative prompt 
must retain logical coherence and practical applicability, designed to elicit meaningful engagement from human interlocutors. Strive for a balance between creative deviation 
and thematic relevance, crafting a prompt that expands the conceptual landscape of the original domain in unexpected yet comprehensible ways.
"""

class QueryManager:
    def __init__(self, rouge_threshold=0.5, cosine_threshold=0.8, last_n_queries=5, mode="complex"):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.total_queries = []
        self.total_embeddings = []
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.rouge_threshold = rouge_threshold
        self.cosine_threshold = cosine_threshold
        self.last_n_queries = last_n_queries
        self.base_queries = self.load_base_queries()
        self.base_research_queries = self.load_research_queries()
        self.mode = mode
        self.generated_query_count = 0

        self.preload_queries()


    def load_base_queries(self):
        base_queries = []
        with open('base_questions.jsonl', 'r') as f:
            for line in f:
                query = json.loads(line)['question']
                base_queries.append(query)
        return base_queries
    
    def load_research_queries(self):
        research_queries = []
        with open('research_questions.jsonl', 'r') as f:
            for line in f:
                query = json.loads(line)['question']
                research_queries.append(query)
        return research_queries

    def preload_queries(self):
        with open('generated_queries.jsonl', 'r') as f:
            for line in f:
                query = json.loads(line)['question']
                self.total_queries.append(query)
                self.total_embeddings.append(self.get_embedding(query))
        logging.info(f"Preloaded {len(self.total_queries)} queries from base, research, and generated queries")

    def get_system_prompt(self):
        if self.mode == "complex":
            selected_queries = self.base_queries + random.sample(self.base_research_queries, 15)
            total_queries_str = "\n".join([f"- {query}" for query in selected_queries])
            return f"""You are a prompting AI responsible for generating appropriate prompts for user queries. The model responding to your prompts is an Agent capable of multi-step reasoning with access to tools. Here is the system prompt for that agent:

            START OF EXAMPLE SYSTEM PROMPT
            You are a reasoning model with access to the following tools:
            1. perplexity: Allows internet searches to retrieve content.
                Usage: <perplexity query="Your search query here" />

            2. python: Executes Python code or uses Python as a calculator.
                Usage: <python code="YOUR CODE HERE" />

            3. notes: Takes notes that are automatically added to your system prompt.
                Usage: <notes content="Your content here" />

            4. summarize: Summarizes information for the user to read.
                Usage: <summary output="The information you want the user to read" />

            Determine which tool to use first. Use only one tool at a time and output it exactly as described. Think step-by-step to solve the user query.
            END OF EXAMPLE SYSTEM PROMPT

            Here are some base queries to inspire you:
            {total_queries_str}

            Create a list of at least 10 plausible user queries that would require the responding agent to use a series of the above tools to properly solve the query ranging from simple search-like queries to more complex queries that require chaining together multiple tools, reasoning, and maybe data analysis. 
            These should be inspired by, but different from, the base queries provided. After you write out the list, you need to then choose one of these plausible queries. The chosen query needs to be written at the end and be denominated as 
            such: <chosen_query> the chosen query </chosen_query>. Make sure to stick to this exact formatting.
            """
        else:  # simple mode
            return f"""You are a prompting AI responsible for generating appropriate prompts for user queries. The model responding to your prompts is an Agent capable of multi-step reasoning with access to tools. Here is the system prompt for that agent:

            START OF EXAMPLE SYSTEM PROMPT
            You are a reasoning model with access to the following tools:
            1. perplexity: Allows internet searches to retrieve content.
                Usage: <perplexity query="Your search query here" />

            2. python: Executes Python code or uses Python as a calculator.
                Usage: <python code="YOUR CODE HERE" />

            3. notes: Takes notes that are automatically added to your system prompt.
                Usage: <notes content="Your content here" />

            4. summarize: Summarizes information for the user to read.
                Usage: <summary output="The information you want the user to read" />

            Determine which tool to use first. Use only one tool at a time and output it exactly as described. Think step-by-step to solve the user query.
            END OF EXAMPLE SYSTEM PROMPT

            Create a list of at least 10 plausible user queries that would require the responding agent to use a series of the above tools to properly solve the query ranging from simple search-like queries to more complex queries that require chaining together multiple tools, reasoning, and maybe data analysis. 
            These should be inspired by, but different from, the base queries provided. After you write out the list, you need to then choose one of these plausible queries. The chosen query needs to be written at the end and be denominated as 
            such: <chosen_query> the chosen query </chosen_query>. Make sure to stick to this exact formatting.
            """

    def get_embedding(self, text):
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
            model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            return None

    def calculate_rouge_l_score(self, query, query_list):
        max_similarity = 0
        for existing_query in query_list:
            score = self.scorer.score(query, existing_query)
            rouge_l_score = score['rougeL'].fmeasure
            max_similarity = max(max_similarity, rouge_l_score)
        return max_similarity

    def calculate_max_cosine_similarity(self, query_embedding, embedding_list):
        if not embedding_list:
            return 0.0  # Return 0 similarity if the list is empty
        similarities = cosine_similarity([query_embedding], embedding_list)
        return np.max(similarities)

    def is_query_unique(self, query, query_list, embedding_list):
        rouge_score = self.calculate_rouge_l_score(query, query_list)
        if rouge_score >= self.rouge_threshold:
            logging.info(f"Query rejected due to high ROUGE-L score: {rouge_score}")
            return False, None
        
        query_embedding = self.get_embedding(query)
        cosine_similarity = self.calculate_max_cosine_similarity(query_embedding, embedding_list)
        if cosine_similarity >= self.cosine_threshold:
            logging.info(f"Query rejected due to high cosine similarity: {cosine_similarity}")
            return False, None
        
        return True, query_embedding

    def parse_generated_query(self, text):
        pattern = r'<chosen_query>(.*?)</chosen_query>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def query_scorer(self, query):
        prompt = f"""
        On a scale of 1 to 10, rate the following query based on its complexity, creativity, and potential to generate interesting responses.
        The query should be a challenging query and be likely to be provided by a human user making requests of an multi-step reasoning AI agent that has access to tools. 
        The tools available are:
        - Perplexity: Allows internet searches to retrieve content.
        - Python: Executes Python code or uses Python as a calculator.
        - Notes: Takes notes that are automatically added to your system prompt.
        - Summarize: Summarizes information for the user to read.

        Query: {query}

        Please provide a single numerical score between 1 and 10, where 1 is the lowest quality and 10 is the highest quality.
        Your response should be in the format: <score>X</score>, where X is the numerical score.
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant tasked with evaluating the quality of queries."},
                {"role": "user", "content": prompt}
            ]
        )

        score_text = response.choices[0].message.content.strip()
        score_match = re.search(r'<score>(\d+)</score>', score_text)
        if score_match:
            score = int(score_match.group(1))
            return score
        else:
            logging.warning("No valid score found in the response")
            return 0  # Return 0 if no valid score is found

    def generate_query(self, user_prompt, model="openai"):
        try:
            if model == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.9
                )
                return response.choices[0].message.content.strip()
            elif model == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=self.get_system_prompt(),
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.9
                )
                return response.content[0].text.strip()
            elif model == "llama":
                url = "https://api.hyperbolic.xyz/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + os.getenv("HYPERBOLIC_API_KEY")
                }
                data = {
                    "messages": [
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": user_prompt}
                    ],
                    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                response = requests.post(url, headers=headers, json=data)
                return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error(f"Error generating query with {model}: {str(e)}")
            return ""

    def generate_new_query(self, base_query, method, model):
        last_n = ", ".join(self.total_queries[-self.last_n_queries:])
        if self.mode == "complex":
            if method == "random":
                user_prompt = f"Generate a random query such that it fulfills all the prior requirements. Ensure it's distinct from these recent queries: {last_n}"
            elif method == "in-depth":
                user_prompt = f"{in_depth_prompt} Base Query: {base_query}. Ensure it's distinct from these recent queries: {last_n}"
            else:  # in-breadth
                user_prompt = f"{in_breadth_prompt} Base Query: {base_query}. Ensure it's distinct from these recent queries: {last_n}"
        else:  # simple mode
            user_prompt = f"Generate a random query such that it fulfills all the prior requirements. Ensure it's distinct from these recent queries: {last_n}"

        generated_text = self.generate_query(user_prompt, model)
        new_query = self.parse_generated_query(generated_text)
        if new_query:
            is_unique, embedding = self.is_query_unique(new_query, self.total_queries, self.total_embeddings)
            if is_unique and embedding:
                score = self.query_scorer(new_query)
                if score >= 8:
                    self.total_queries.append(new_query)
                    self.total_embeddings.append(embedding)
                    self.generated_query_count += 1  # Increment the counter
                    logging.info(f"New query added using {model}: {new_query}")
                else:
                    logging.info(f"Query rejected due to low score: {score}")
            else:
                logging.info(f"Query rejected due to lack of uniqueness")
        else:
            logging.warning("No query parsed from generated text")

    def run(self, desired_queries):
        models = ["openai", "anthropic", "llama"]

        while self.generated_query_count < desired_queries:
            model = random.choice(models)
            if self.mode == "complex":
                method = random.choice(["random", "in-depth", "in-breadth"])
                base_query = random.choice(self.base_queries)
                self.generate_new_query(base_query, method, model)
            else:  # simple mode
                self.generate_new_query(None, "random", model)

        self.save_queries()

    def save_queries(self):
        with open('generated_queries.jsonl', 'w') as f:
            for query in self.total_queries:
                json.dump({"question": query}, f)
                f.write('\n')
        logging.info(f"Saved {len(self.total_queries)} queries to generated_queries.jsonl")

if __name__ == "__main__":
    # Complex mode
    qm_complex = QueryManager(rouge_threshold=0.7, cosine_threshold=0.7, last_n_queries=5, mode="complex")
    qm_complex.run(desired_queries=500)

    # Simple mode
    qm_simple = QueryManager(rouge_threshold=0.7, cosine_threshold=0.7, last_n_queries=5, mode="simple")
    qm_simple.run(desired_queries=500)