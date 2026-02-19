import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.embeddings.create(
    input="hello world",
    model="text-embedding-3-small"
)

print(response.data[0].embedding[:5])  # should print 5 float values