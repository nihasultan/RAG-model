from retrieval import hybrid_search

test_queries = [
    {"query": "What is the objective?", "expected": "objective"},
    {"query": "How is deployment handled?", "expected": "deployment"},
]

correct = 0

for t in test_queries:
    results = hybrid_search(t["query"], k=3)
    texts = " ".join([r["text"].lower() for r in results])

    if t["expected"] in texts:
        correct += 1

accuracy = correct / len(test_queries)

print(f"Correct: {correct}/{len(test_queries)}")
print(f"Accuracy: {accuracy * 100:.2f}%")