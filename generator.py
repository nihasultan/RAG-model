from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
)

def generate_answer(query, docs):
    context = "\n\n".join([d.get("text", "") for d in docs])

    prompt = f"""
You are an expert assistant.

Answer the question using ONLY the context below.

Write the answer in clear bullet points.

Context:
{context}

Question:
{query}

Answer:
"""
    result = generator(
        prompt,
        max_new_tokens=300,
        do_sample=False
    )

    text = result[0]["generated_text"]

    lines = text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip("- ").strip()
        if line:
            cleaned.append(f"- {line}")

    return "\n".join(cleaned)