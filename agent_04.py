# agent with goals and tools

import os, sys
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.models.ollama import Ollama

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.calculator import CalculatorTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.openweather import OpenWeatherTools

load_dotenv()
api_key = (os.getenv('OPENAI_API_KEY') or
           sys.exit('file .env does not contain OPENAI_API_KEY'))
id_model = os.getenv('ID_MODEL') or sys.exit('file .env does not contain ID_MODEL')

agent = Agent(# model=OpenRouter(id=id_model),
              model=Ollama(id="llama3.2"),
              description='You respond using actual data from the Internet',
              tools=[
                  GoogleSearchTools(),
                  DuckDuckGoTools(),
                  Newspaper4kTools(),
                  CalculatorTools(),
                  YFinanceTools()
              ],
              debug_mode=True,
              debug_level=1)

if __name__ == '__main__':
    while question := input('User: ').strip():
        response = agent.run(question)
        print(f'Agent: {response.content}')
