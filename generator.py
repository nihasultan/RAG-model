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
    Context:
    {context}

    Question: {query}

    Answer briefly and clearly.
    """

    response = generator(prompt)[0]["generated_text"]

    return response.strip()