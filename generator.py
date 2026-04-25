from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
)

def generate_answer(query, docs):
    context = "\n\n".join(
        [d.get("text", "")[:300] for d in docs[:3]]  
    )

    prompt = f"""Answer the question using the context below. Write 4 to 6 detailed bullet points. Do not cut off mid-sentence.

Context:
{context}

Question:
{query}

Answer:"""

    result = generator(
        prompt,
        max_new_tokens=300,   
        min_new_tokens=80,    
        do_sample=False,
        no_repeat_ngram_size=3, 
    )

    text = result[0]["generated_text"].strip()

    lines = [line.strip().lstrip("-•* ").strip() for line in text.split("\n")]
    cleaned = [f"- {line}" for line in lines if line]

    return "\n\n".join(cleaned)  