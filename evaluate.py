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

def evaluate_answer(row):
    system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
    prompt = f"I want you to evaluate a predicted answer. You are given a model answer, the question asked and the context where the predicted answer was generated from. Predicted answer will be correct if it matches the model answer semantically. Return 1 if the predicted answer is correct and 0 if it is wrong. Strictly only return 1 or 0.\nThe question:{row['question']}\nThe context:{row['context']}\nThe model answer: {row['answer']}\nThe predicted answer: {row['predicted_answers']}"
    return get_response(system_prompt, prompt)

df = pd.read_csv("predicted_answers.csv",index_col=0)
df["evaluation"] = df.apply(evaluate_answer, axis=1)
df.to_csv("evaluation.csv")

