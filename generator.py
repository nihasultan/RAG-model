from transformers import pipeline

generator = pipeline(
    "text2text-generation",  
    model="flan-t5-large"
)

def generate_answer(query, docs):
    context = "\n".join([d.get("text", "") for d in docs])

    prompt = f"""
Answer the question in detail using the context below.

Write at least 5 bullet points.

Context:
{context}

Question:
{query}

Detailed Answer:
"""

    result = generator(
    prompt,
    max_length=800,
    min_length=100  
)

    return result[0]["generated_text"]