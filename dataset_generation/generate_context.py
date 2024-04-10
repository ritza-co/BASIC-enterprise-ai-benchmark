import json
import os
import openai
from dotenv import load_dotenv

# TODO:automate all scripts in one script
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))


def get_response(system_prompt, user_input):
	messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
	model = "gpt-4-0125-preview"
	completion = client.chat.completions.create(model=model, messages=messages)
	return completion.choices[0].message.content


questions = ["Hi there, do you provide dental coverage as well?",
             "Do I need to pay for the complete year or are there any installment plans?",
             "Can I make adjustments to my plan midway?",
             "Hi, I'm moving to another state. Does my coverage remain the same?",
             "How to file a complaint against improper service?",
             "Can I track my claim status online?",
             "What's the procedure to transfer my policy to a different insurer?",
             "I'm not happy with my current plan. Can I switch between health insurance plans?",
             "Is there any option to get paperless documents?",
             "Why has my claim been rejected?",
             "How can I get a copy of all my claim history?",
             "I don't understand my policy statement, who can I ask for help?",
             "Can I extend the coverage to my spouse under same plan?",
             "Can I get a plan which covers out of country medical expenses?",
             "not being able to pay online, any alternate options?",
             "Any discount available for senior citizens?",
             "Where can I find list of network hospitals?",
             "Is vision care included in policy?",
             "Wanna know if ambulance services are covered?",
             "Can I opt for additional riders with my existing policy?"
]

system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"

data = []

for question in questions:
	user_prompt = f"""Please create 2 - 3 paragraphs of context for the following question:

    {question}

    The context paragraphs are the paragraphs from which this question can be answered. The context paragraphs are not the answers, but one may find an answer to the above question in it. The context paragraphs should be difficult to extract answers from. The context paragraphs should be like an information extracted from a guide or a manual or a PDF. The context paragraphs should sound very enterprise-like. The context paragraphs should contain information that can provide an answer or partial answer to the above question. The answer to the question can be very unclear, or hidden amongst information that is not directly relevant. The context paragraphs should not sound like a comprehension or a story. The context paragraph should have other information as well which should be related to the topic of question but does not answer the question.
    """
	context = get_response(system_prompt, user_prompt)

	print(f"Question: {question}")
	print("\n")
	print(f"Context: \n{context}")

	data.append({'question': question, 'answer': None, 'context': context})

	with open('qa_dataset.json', 'w') as file:
		json.dump(data, file)
