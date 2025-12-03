import os, sys
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder

# локальный эмбеддер Agno на базе SentenceTransformers
embedder = SentenceTransformerEmbedder(id='all-MiniLM-L6-v2')

text = "Все коты падают на землю с одинаковым ускорением."
# text = 'Как загрузить файл в базу знаний Agno?'
vector1 = embedder.get_embedding(text)
print(f'Vector dimension: {len(vector1)}')
print(f'First components {vector1[:10]}')


text = "Коты с разной массой падают на землю с одинаковым ускорением."
# text = "Как добавить документ в базу знаний Agno?"
vector2 = embedder.get_embedding(text)

print(f'Vector dimension: {len(vector2)}')
print(f'First components: {vector2[:10]}')

sum2 = 0
for i in range(len(vector1)):
    sum2 += vector1[i] * vector2[i]
print(f'scalar product: {sum2}')
