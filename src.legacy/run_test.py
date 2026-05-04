import sys
sys.path.insert(0, "/home/tianrenu/projects/dachuang_project/RAGarden/src")

from agentic_rag.pipeline import run, TEST_QUESTIONS

for question in TEST_QUESTIONS:
    print(f"\n{'='*50}")
    print(f"Q: {question}")
    print("-" * 50)
    try:
        result = run(question)
        print(f"[type] {result['router_type']}  [Q] {result['Q']:.3f}  [attempts] {result['attempts']}")
        print(f"[A] {result['answer'][:200]}")
    except Exception as e:
        print(f"[ERROR] {e}")
