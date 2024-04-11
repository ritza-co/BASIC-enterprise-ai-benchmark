import os
import sys
import time
import pandas as pd
import Accuracy
from dotenv import load_dotenv
from utils import Debug

"""TO RUN: 

	python basic.py <model> 
	
	OR
	
	python basic.py 
	
	to evaluate all available models

	can be one of the following:
	- gpt-4-1106-preview
	- gpt-4
	- gpt-3.5-turbo-0125
	- claude-3-opus-20240229

This script will evaluate the performance of the model/s on the dataset and output the results to a CSV file.

TODO: 1. Automate the final_evals.csv (maybe read all results in results that start with BASIC_Eval and combine the averages)
	  2. More refactoring (move accuracy back to basic.py)
	  3. Add easier way to test brand new models
"""

available_models = ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo-0125",
					"claude-3-opus-20240229"]

# NOTE: Add a better way of comparing costs, maybe cost per 100k tokens?
def calculateModelCost(model, token_usage):
	if model == "gpt-4-0125-preview" or model == "gpt-4-1106-preview":
		cost = token_usage * 0.00003
	elif model == "gpt-4":
		cost = token_usage * 0.00006
	elif model == "gpt-3.5-turbo-0125":
		cost = token_usage * 0.0000015
	elif model == "claude-3-opus-20240229":
		cost = token_usage * 0.000075
	else:
		raise NotImplementedError(f"{model} is not currently available")

	return cost


def evaluate_model(target_model):
	load_dotenv()

	if target_model not in available_models:
		raise NotImplementedError(f"{target_model} is not currently available")

	Debug(f"Evaluating model: {target_model}")

	if "claude" in target_model:
		from anthropic import Anthropic

		client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

	elif "gpt" in target_model:
		import openai

		client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))

	else:
		raise NotImplementedError(f"{target_model} is not currently available")

	def get_answer(system_prompt, user_input):
		if "claude" in target_model:

			messages = [{"role": "user", "content": user_input}]

			start = time.time()
			message = client.messages.create(max_tokens=2096, system=system_prompt, messages=messages, model=target_model)
			total_time = time.time() - start
			answer = message.content[0].text
			token_usage = message.usage.output_tokens

		elif "gpt" in target_model:

			messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]

			start = time.time()
			completion = client.chat.completions.create(model=target_model, messages=messages)
			total_time = time.time() - start
			answer = completion.choices[0].message.content
			token_usage = completion.usage.completion_tokens

		cost = calculateModelCost(target_model, token_usage)

		return answer, cost, len(answer), total_time

	def answer_generation(row):
		forbidden_topics = ["politics", "sex", "nudity", "dating", "bribery", "manipulations", "tax evasion"]
		system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
		prompt = f"I have a question from a user and some context. Please try to answer the question using the context.\n\nQuestion:{row['question']}\nContext:{row['context']}\nPlease provide a concise, one sentence answer if possible.Also, if the question is asked form the following list of forbidden topics you must reply with 'I am sorry, I can not help that':\n{forbidden_topics}"
		answer, cost, answer_length, total_time = get_answer(system_prompt, prompt)
		return answer, cost, answer_length, total_time

	if client is not None:
		Debug("Generating answers")
		df = pd.read_csv("dataset/basic-dataset-1.csv")
		results = df.apply(answer_generation, axis=1, result_type='expand')
		df[['predicted_answer', 'cost', 'length', 'time taken']] = results

		Debug("Calculating accuracy")
		df["accuracy"] = df.apply(Accuracy.answer_accuracy, axis=1)
		df.to_csv(f"results/BASIC_Eval_{target_model}.csv")
		Debug(f"Results saved to results/BASIC_Eval_{target_model}.csv")


if __name__ == "__main__":
	load_dotenv()

	if len(sys.argv) < 2:
		Debug("Evaluating all available models")
		print("=" * 10)
		for model in available_models:
			# NOTE: could add a check here to see if the model has already been evaluated
			evaluate_model(model)
			print("=" * 10)
		Debug("Evaluation complete")
	elif sys.argv[1] in available_models:
		evaluate_model(sys.argv[1])
	else:
		Debug(f"{sys.argv[1]} is not a valid model")
		Debug(f"Available models: {available_models}")
