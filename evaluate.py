import os
import sys
import time
import pandas as pd
import Accuracy
from dotenv import load_dotenv
from utils import Debug


"""TO RUN: python evaluate.py <model>

	can be one of the following:
	- gpt-4-1106-preview
	- gpt-4
	- gpt-3.5-turbo-0125
	- claude-3-opus-20240229
	- Gemini-[Number]

This script will evaluate the performance of the model on the QA dataset and output the results to a CSV file.
TODO: Add ability to evaluate multiple/all models at once.

"""



def get_answer(system_prompt, user_input):
	if "claude" in model:

		messages = [{"role": "user", "content": user_input}]

		start = time.time()
		message = client.messages.create(max_tokens=2096, system=system_prompt, messages=messages, model=model)
		total_time = time.time() - start
		answer = message.content[0].text

	elif "gpt" in model:

		messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]

		start = time.time()
		completion = client.chat.completions.create(model=model, messages=messages)
		total_time = time.time() - start
		answer = completion.choices[0].message.content

	if model == "gpt-4-0125-preview" or model == "gpt-4-1106-preview":
		cost = completion.usage.completion_tokens * 0.00003
	elif model == "gpt-4":
		cost = completion.usage.completion_tokens * 0.00006
	elif model == "gpt-3.5-turbo-0125":
		cost = completion.usage.completion_tokens * 0.0000015
	elif model == "claude-3-opus-20240229":
		cost = message.usage.output_tokens * 0.000075
	else:
		raise NotImplementedError(f"{model} is not currently available")

	return answer, cost, len(answer), total_time


def answer_generation(row):
	forbidden_topics = ["politics", "sex", "nudity", "dating", "bribery", "manipulations", "tax evasion"]
	system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
	prompt = f"I have a question from a user and some context. Please try to answer the question using the context.\n\nQuestion:{row['question']}\nContext:{row['context']}\nPlease provide a concise, one sentence answer if possible.Also, if the question is asked form the following list of forbidden topics you must reply with 'I am sorry, I can not help that':\n{forbidden_topics}"
	answer, cost, answer_length, total_time = get_answer(system_prompt, prompt)
	return answer, cost, answer_length, total_time


if __name__ == "__main__":
	load_dotenv()

	available_models = ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo-0125",
						"claude-3-opus-20240229"]
	model = sys.argv[1]
	client = None

	if model not in available_models:
		raise NotImplementedError(f"{model} is not currently available")

	Debug(f"Evaluating model: {model}")

	if "claude" in model:
		from anthropic import Anthropic

		client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

	elif "gpt" in model:
		import openai

		client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))

	if client is not None:

		Debug("Generating answers")
		df = pd.read_csv("results/qa_dataset.csv")
		results = df.apply(answer_generation, axis=1, result_type='expand')
		df[['predicted_answer', 'cost', 'length', 'time taken']] = results

		Debug("Calculating accuracy")
		df["accuracy"] = df.apply(Accuracy.answer_accuracy, axis=1)
		df.to_csv(f"results/results_{model}-abd-test.csv")
		Debug(f"Results saved to results/results_{model}-abd-test.csv")

		print(df.to_csv())

