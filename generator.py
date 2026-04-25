from transformers import pipeline

# Load once (cached by Streamlit automatically)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # 🔥 small = fast + deployable
)

def generate_answer(query, docs):
    context = "\n\n".join([d.get("text", "") for d in docs])

    prompt = f"""
Answer the question in detail using the context below.

Give at least 5 bullet points.

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=500,
        do_sample=False
    )

    return result[0]["generated_text"]