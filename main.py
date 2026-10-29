import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
chat_completion = client.chat.completions.create(
    messages=[
        {"role":"user", "content":"Tell me a joke about Bangladesh!"}
    ],
    model="gpt-5-mini"
)

print(chat_completion.choices[0].message.content)

