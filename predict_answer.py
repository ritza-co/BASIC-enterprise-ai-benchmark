import os

import openai
from dotenv import load_dotenv
import pandas as pd

import json

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))

def get_response(system_prompt, user_input):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append(
        {"role": "user", "content": user_input},
    )
    model = "gpt-4-0125-preview"
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content

def answer_generation(row):
    system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
    prompt = f"I have a question from a user and some context. Please try to answer the question using the context.\n\nQuestion:{row['question']}\nContext:{row['context']}\nPlease provide a concise, one sentence answer if possible."
    return get_response(system_prompt, prompt)

df = pd.read_csv("qa_dataset.csv")
df["predicted_answers"] = df.apply(answer_generation, axis=1)
df.to_csv("predicted_answers.csv")

