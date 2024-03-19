import os

import openai
from dotenv import load_dotenv


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))

def get_response(system_prompt, user_input):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append(
        {"role": "user", "content": user_input},
    )
    model = "gpt-4"
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content

questions = []

system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"

user_prompt = """I am building a dataset to benchmark chatbots. I need questions that simulate a variety of questions that an end user might ask a chatbot on a health insurance site. These should be varied between different stages of the funnel - for example, questions a potential new user might ask about signing up, buying policies, and comparing options, as well as questions from existing users who would be asking for more specific things like making a claim, managing their account, or other customer support. Here are the questions I already have: {}.

Please output 50 new questions, not repeating any of the above, one per line. Make the questions realistic. So the user might add irrelevant context, make typing errors, use bad grammar. Some questions should be 2-3 sentences long, some should just be a few words. They should be differently formatted and in different styles as if coming from different users, some with salutation, some without. Some formal, some informal. Remember, include typos and some very short questions of only two or three keywords""".format('\n'.join(questions))

response = get_response(system_prompt, user_prompt)
for q in response.split("\n"):
    print(q)









