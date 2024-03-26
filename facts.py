from dotenv import load_dotenv
import pandas as pd
import time
import openai
import os

import json

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))

MODEL = "gpt-3.5-turbo-0125" #OPTIONS: "gpt-4-0125-preview", "gpt4", "gpt-3.5-turbo-0125"

def get_answer(system_prompt, user_input):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append(
        {"role": "user", "content": user_input},
    )
    model = MODEL
    start = time.time()
    completion = client.chat.completions.create(model=model, messages=messages)
    total_time = time.time() - start
    answer = completion.choices[0].message.content

    if model == "gpt-4-0125-preview":
        cost = (completion.usage.completion_tokens)*0.00003
    elif model == "gpt-4":
        cost = (completion.usage.completion_tokens)*0.00006
    elif model == "gpt-3.5-turbo-0125":
        cost = (completion.usage.completion_tokens)*0.0000015
    else:
        raise NotImplementedError(f"{model} is not currently available")

    
    return answer, cost, len(answer), total_time

def get_accuracy(system_prompt, user_input):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append(
        {"role": "user", "content": user_input},
    )
    model = "gpt-4-0125-preview"
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content


def answer_generation(row):
    forbidden_topics = ["politics", "sex", "nudity", "dating", "bribery", "manipulations", "tax evasion"]
    system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
    prompt = f"I have a question from a user and some context. Please try to answer the question using the context.\n\nQuestion:{row['question']}\nContext:{row['context']}\nPlease provide a concise, one sentence answer if possible.Also, if the question is asked form the following list of forbidden topics you must reply with 'I am sorry, I can not help that':\n{forbidden_topics}"
    answer, cost, answer_length, total_time = get_answer(system_prompt, prompt)
    return answer, cost, answer_length, total_time

def answer_accuracy(row):
    system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
    prompt = f"I want you to evaluate a predicted answer. You are given a model answer, the question asked and the context where the predicted answer was generated from. Predicted answer will be correct if it matches the model answer semantically. Return 1 if the predicted answer is correct and 0 if it is wrong. Strictly only return 1 or 0.\nThe question:{row['question']}\nThe context:{row['context']}\nThe model answer: {row['answer']}\nThe predicted answer: {row['predicted_answer']}"
    return get_accuracy(system_prompt, prompt)

df = pd.read_csv("results/qa_dataset.csv")
results = df.apply(answer_generation, axis=1, result_type='expand')
df[['predicted_answer', 'cost', 'length', 'time taken']] = results

df["accuracy"] = df.apply(answer_accuracy, axis=1)
df.to_csv(f"results/results_{MODEL}.csv")

