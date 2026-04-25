from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=300
)

def generate_answer(query, docs):
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = generator(prompt)[0]["generated_text"]
    return response