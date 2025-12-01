# agent with added knowledge

import os, sys
from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from dotenv import load_dotenv
# from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.ollama import Ollama


load_dotenv()
api_key = (os.getenv('OPENAI_API_KEY') or
           sys.exit('file .env does not contain OPENAI_API_KEY'))
id_model = os.getenv('ID_MODEL') or sys.exit('file .env does not contain ID_MODEL')

# локальный эмбеддер Agno на базе SentenceTransformers
embedder = SentenceTransformerEmbedder(id='all-MiniLM-L6-v2')
# embedder = OpenAIEmbedder(id="text-embedding-3-small")

# локальная векторная база LanceDB с гибридным поиском
vector_db = LanceDb(
    table_name='text_documents',  # название таблицы для эмбеддингов
    uri='lancedb_storage',  # локальная папка для LanceDb
    search_type=SearchType.hybrid,  # гибридный поиск: по смыслу и ключевым словам
    embedder=embedder  # подключение эмбеддера
)

#  создаем knowledge, объект базы знаний
knowledge = Knowledge(vector_db=vector_db)
knowledge.add_content(path='aliens.txt')

# агент со знаниями
agent = Agent(
    model=OpenRouter(id=id_model, api_key=api_key),  # подключение модели
    # model=Ollama(id="llama3.2"),
    knowledge=knowledge,  # подключаем базу знаний
    search_knowledge=True, # разрешаем ее использование
    debug_mode=True  # выключаем режим отладки (по умолчанию)
)

# question = 'В каком городе проходила первая конференция Agno? сколько стран в ней участвовало?'
question = input('User: ').strip()


response = agent.run(question)
print(f'Agent: {response.content}')