# agent with memory

import os, sys
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv


load_dotenv()
api_key = (os.getenv('OPENAI_API_KEY') or
           sys.exit('file .env does not contain OPENAI_API_KEY'))
id_model = os.getenv('ID_MODEL') or sys.exit('file .env does not contain ID_MODEL')

# память агента в Sqlite
db = SqliteDb(db_file='data.db')

agent = Agent(
    model=OpenRouter(id=id_model),
    session_id = 'dialog', # Deutsch
    db=db,
    add_history_to_context=True,
    num_history_runs=0,
)

if __name__ == '__main__':
    while question := input('User: ').strip():
        response = agent.run(question)
        print(f'AI: {response.content}')