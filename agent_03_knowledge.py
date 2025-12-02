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

embedder = SentenceTransformerEmbedder(id='all-MiniLM-L6-v2')
# embedder = OpenAIEmbedder(id="text-embedding-3-small")
# embedder = SentenceTransformerEmbedder(id='multilingual-e5-large')  # all-mpnet-base-v2


vector_db = LanceDb(
    table_name='text_documents',
    uri='lancedb_storage',
    search_type=SearchType.hybrid,
    embedder=embedder
)

knowledge = Knowledge(vector_db=vector_db)
knowledge.add_content(
    path="knowledge/GrKRFCH4-01.pdf",
    skip_if_exists=True
)

# агент со знаниями
agent = Agent(
    # model=OpenRouter(id=id_model, api_key=api_key),
    model=Ollama(id="llama3.2"),
    knowledge=knowledge,
    search_knowledge=True,
    debug_mode=True
)


while query := input('User: ').strip():
    response = agent.run(query)
    print(f'Agent: {response.content}')