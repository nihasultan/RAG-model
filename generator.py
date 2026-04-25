from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, retrieved_docs):
    context = "\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}

Give a detailed answer in 5-6 points.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content