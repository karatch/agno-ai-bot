import os, sys
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY') or sys.exit('file .env does not contain OPENAI_API_KEY')
id_model = os.getenv('ID_MODEL') or sys.exit('file .env does not contain ID_MODEL')

agent = Agent(model=OpenRouter(id=id_model))

while question := input('User: '):
    response = agent.run(question)
    print(f'AI: {response.content}')