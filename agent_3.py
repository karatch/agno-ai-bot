# agent with knowledge

import os, sys
from agno.agent import Agent
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv


load_dotenv()
api_key = (os.getenv('OPENAI_API_KEY') or
           sys.exit('file .env does not contain OPENAI_API_KEY'))
id_model = os.getenv('ID_MODEL') or sys.exit('file .env does not contain ID_MODEL')

# локальный эмбеддер Agno на базе SentenceTransformers
embedder = SentenceTransformerEmbedder(id='all-MiniLM-L6-v2')