from transformers import pipeline

generator = pipeline(
    "text2text-generation",  
    model="google/flan-t5-base"
)

def generate_answer(query, docs):
    context = "\n".join([d.get("text", "") for d in docs])

    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question in detail.

Instructions:
- Give a complete and structured answer
- Use bullet points if possible
- Explain clearly
- Do NOT repeat the question
- If context is insufficient, say "Not enough information"

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(prompt, max_length=512)

    return result[0]["generated_text"]