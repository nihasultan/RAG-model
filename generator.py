from transformers import pipeline

generator = pipeline(
    "text2text-generation",  
    model="google/flan-t5-base",
    max_new_tokens=150,
    do_sample=False
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
    - Do NOT say "based on context"
    - If context is insufficient, say "Not enough information"

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

     response = client.chat.completions.create(
        model="google/flan-t5-base",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,   
        temperature=0.3   
    )

    return response.choices[0].message.content