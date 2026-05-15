# assignments_06/warmup_06.py

"""
Week 6 Warmup Assignment
Topics:
- Prompt engineering, fine-tuning, and RAG
- Keyword-based RAG
- Semantic RAG
- LlamaIndex concepts

Note:
The Brightleaf PDF folder is missing from my local repo, so the LlamaIndex
Brightleaf section includes safe error handling.
"""

from dotenv import load_dotenv
import os
import string
from pathlib import Path

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("OPENAI_API_KEY found.")
else:
    print("OPENAI_API_KEY not found. Some LlamaIndex sections may not run.")


# ------------------------------------------------------------
# RAG Concepts
# ------------------------------------------------------------

# --- Concepts Question 1 ---
"""
Scenario A:
Best approach: RAG.
Reason: The legal team has hundreds of internal PDFs updated every quarter.
RAG is best because it can retrieve answers from the most current documents.

Scenario B:
Best approach: Fine-tuning.
Reason: The startup wants a very specific brand voice and has 3,000 examples.
Fine-tuning can help the model learn that writing style.

Scenario C:
Best approach: Prompt engineering.
Reason: The analyst only needs questions answered from one short two-page report.
A simple prompt with the report included is enough.
"""

# --- Concepts Question 2 ---
"""
A confidently wrong answer is more harmful than an answer that says "I am not sure"
because people may trust confident language and act on false information.

Example:
If an AI confidently gives the wrong medical instruction or wrong legal policy,
someone could make a serious decision based on incorrect information.

Tone matters because confident wording makes an answer feel more reliable, even
when it is not supported by facts.
"""

# --- Concepts Question 3 ---
"""
Correct RAG pipeline order:

1. Extract text from source documents
   - Read the text from PDFs, text files, or other source documents.

2. Split text into chunks
   - Break large documents into smaller pieces.

3. Convert text chunks into embeddings
   - Turn each chunk into a list of numbers that represents meaning.

4. Receive the user's query
   - The user asks a question.

5. Embed the user's query
   - Turn the user's question into a list of numbers too.

6. Retrieve the most relevant chunks
   - Find the document chunks closest in meaning to the question.

7. Inject retrieved chunks into the prompt
   - Put the retrieved evidence into the prompt.

8. Generate a response from the LLM
   - The model answers using the retrieved context.
"""


# ------------------------------------------------------------
# Keyword RAG
# ------------------------------------------------------------

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }

    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }

        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)

    best = next(((name, content) for score, name, content in scores if score > 0), None)

    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


# --- Keyword Question 1 ---
print("\n" + "=" * 70)
print("Keyword Question 1")

query = "What are your hours on the weekend?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

"""
Comment:
The selected document should be hours.txt because the query contains "hours"
and "weekend," and the hours document contains weekend opening and closing times.
"""


# --- Keyword Question 2 ---
print("\n" + "=" * 70)
print("Keyword Question 2")

query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

"""
Comment:
Keyword RAG may not get this right because the menu document has coffee drinks
but may not contain the exact word "caffeine." This shows a limitation of keyword
retrieval: it depends on exact word overlap. Semantic retrieval would do better
because it can understand related meaning, such as caffeine-free or decaf.
"""


# --- Keyword Question 3 ---
print("\n" + "=" * 70)
print("Keyword Question 3")

"""
Prediction before running:
I predict loyalty.txt should be selected because "sign up for rewards" is related
to a loyalty program. However, keyword search might fail because the exact word
"rewards" may not appear in loyalty.txt.
"""

query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

"""
Comment after running:
If loyalty.txt was selected, the prediction was correct.
If no document was selected, that shows keyword RAG cannot always understand
synonyms like "rewards" and "loyalty."
"""


# ------------------------------------------------------------
# Semantic RAG Concepts
# ------------------------------------------------------------

# --- Semantic Question 1 ---
"""
A vector embedding is a way to convert text into numbers that represent meaning.
Texts with similar meanings should have similar vectors.

If two chunks have cosine similarity scores of 0.85 and 0.30, the 0.85 chunk
is more relevant. The higher score means it is closer in meaning to the query.

Semantic search can find relevant text even without exact word matches because
it compares meaning, not only matching words.
"""

# --- Semantic Question 2 ---
"""
| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap                | Meaning using embeddings |
| What is retrieved?         | Full document                     | Relevant chunks |
| Can it handle synonyms?    | No                                | Yes, often |
| Storage format             | Plain text dictionary             | Vector store / index |
| Relevance score            | Number of overlapping keywords    | Similarity score |
"""


# ------------------------------------------------------------
# LlamaIndex Section
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LlamaIndex Section")

try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")

    possible_paths = [
        Path("lessons/06_AI_augmentation/brightleaf_pdfs"),
        Path("06_AI_augmentation/brightleaf_pdfs"),
        Path("../lessons/06_AI_augmentation/brightleaf_pdfs"),
        Path("../06_AI_augmentation/brightleaf_pdfs"),
        Path("../../06_AI_augmentation/brightleaf_pdfs"),
    ]

    brightleaf_dir = None
    for path in possible_paths:
        if path.exists():
            brightleaf_dir = path
            break

    if brightleaf_dir is None:
        raise FileNotFoundError(
            "brightleaf_pdfs folder not found. This folder is missing from the local repo."
        )

    print(f"Using Brightleaf PDF directory: {brightleaf_dir}")

    brightleaf_docs = SimpleDirectoryReader(str(brightleaf_dir)).load_data()
    print(f"Loaded {len(brightleaf_docs)} Brightleaf documents.")

    index = VectorStoreIndex.from_documents(brightleaf_docs)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # --- LlamaIndex Question 1 ---
    print("\n" + "=" * 70)
    print("LlamaIndex Question 1")

    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]

    for question in questions:
        print("\nQuestion:", question)
        response = query_engine.query(question)

        print("\nAnswer:")
        print(response)

        print("\nRetrieved source nodes:")
        for i, source_node in enumerate(response.source_nodes, start=1):
            score = source_node.score
            text = source_node.node.get_content().replace("\n", " ")
            print(f"\nSource node {i}")
            print(f"Similarity score: {score}")
            print(f"Chunk preview: {text[:150]}")

    """
    Comment:
    The retrieved chunks should look relevant when they mention benefits, security,
    policies, access, or employee information. If the model has enough context,
    it may sound confident. If the context is weak, it should hedge or say the
    answer is not available.
    """

    # --- LlamaIndex Question 2 ---
    print("\n" + "=" * 70)
    print("LlamaIndex Question 2")

    comparison_question = "What employee benefits does BrightLeaf offer?"

    for top_k in [1, 5]:
        print(f"\nRunning with similarity_top_k={top_k}")
        comparison_engine = index.as_query_engine(similarity_top_k=top_k)
        response = comparison_engine.query(comparison_question)

        print("\nResponse:")
        print(response)

        print("\nSource nodes:")
        for i, source_node in enumerate(response.source_nodes, start=1):
            text = source_node.node.get_content().replace("\n", " ")
            print(f"Node {i} | score={source_node.score} | preview={text[:150]}")

    """
    Comment:
    More context can help when the extra chunks are relevant, but more context is
    not always better. If similarity_top_k is too large, unrelated chunks may make
    the answer less focused.
    """

    # --- LlamaIndex Question 3 ---
    print("\n" + "=" * 70)
    print("LlamaIndex Question 3")

    struggle_question = "What is BrightLeaf's exact company profit for last year?"
    response = query_engine.query(struggle_question)

    print("\nQuestion:", struggle_question)
    print("\nResponse:")
    print(response)

    print("\nRetrieved chunks:")
    for i, source_node in enumerate(response.source_nodes, start=1):
        text = source_node.node.get_content().replace("\n", " ")
        print(f"Node {i} | score={source_node.score} | preview={text[:200]}")

    """
    Comment:
    I expected this question to be hard because exact company profit may not be
    in the documents. A safe RAG system should say the information is not found
    instead of guessing.
    """

    # --- LlamaIndex Question 4 ---
    print("\n" + "=" * 70)
    print("LlamaIndex Question 4")

    judge_llm = OpenAI(model="gpt-4o-mini")
    faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

    good_q = "What employee benefits does BrightLeaf offer?"
    good_response = query_engine.query(good_q)

    bad_q = "What is the CEO's favorite restaurant?"
    bad_response = query_engine.query(bad_q)

    print("\nEvaluating good query:")
    print(good_q)

    good_faithfulness = faithfulness_evaluator.evaluate_response(
        query=good_q,
        response=good_response,
    )
    good_relevancy = relevancy_evaluator.evaluate_response(
        query=good_q,
        response=good_response,
    )

    print("Faithfulness passing:", good_faithfulness.passing)
    print("Faithfulness score:", getattr(good_faithfulness, "score", None))
    print("Faithfulness feedback:", good_faithfulness.feedback)

    print("Relevancy passing:", good_relevancy.passing)
    print("Relevancy score:", getattr(good_relevancy, "score", None))
    print("Relevancy feedback:", good_relevancy.feedback)

    print("\nEvaluating lower-quality query:")
    print(bad_q)

    bad_faithfulness = faithfulness_evaluator.evaluate_response(
        query=bad_q,
        response=bad_response,
    )
    bad_relevancy = relevancy_evaluator.evaluate_response(
        query=bad_q,
        response=bad_response,
    )

    print("Faithfulness passing:", bad_faithfulness.passing)
    print("Faithfulness score:", getattr(bad_faithfulness, "score", None))
    print("Faithfulness feedback:", bad_faithfulness.feedback)

    print("Relevancy passing:", bad_relevancy.passing)
    print("Relevancy score:", getattr(bad_relevancy, "score", None))
    print("Relevancy feedback:", bad_relevancy.feedback)

    """
    Comment:
    A faithfulness score of 1.0 means the response is supported by the retrieved
    context. A score of 0.0 means the response is not supported by the context.

    Relevancy measures whether the answer responds to the question. It is different
    from faithfulness because an answer can be based on the context but still not
    answer the user clearly.

    LLM-as-a-judge means using another LLM to evaluate the answer. It is useful
    because RAG answers are written in natural language, so exact-match scoring
    is not enough.
    """

except Exception as e:
    print("\nLlamaIndex Brightleaf section could not run successfully.")
    print("Reason:")
    print(e)
    print(
        "\nThis likely happened because the brightleaf_pdfs folder is missing from the local repo."
    )
    print(
        "The concepts, keyword RAG, and semantic RAG sections above still run successfully."
    )