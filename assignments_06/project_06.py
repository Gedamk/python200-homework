# assignments_06/project_06.py

"""
Week 6 Mini-Project
Groundwork Coffee Co. Q&A Assistant

Goal:
Build a RAG-powered assistant that answers questions from real Groundwork Coffee documents.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# ------------------------------------------------------------
# Step 1: Setup
# ------------------------------------------------------------

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY not found. Please add it to your .env file."

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print(f"Using document directory: {docs_dir}")


# ------------------------------------------------------------
# Step 2: Load the Documents
# ------------------------------------------------------------

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print("\nLoaded documents:")
print(f"Number of documents loaded: {len(documents)}")

for doc in documents:
    file_name = doc.metadata.get("file_name", "Unknown file")
    print(f"- {file_name}")


# ------------------------------------------------------------
# Step 3: Build the Index and Query Engine
# ------------------------------------------------------------

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def print_top_source(question, response):
    print("\n" + "=" * 80)
    print("Question:")
    print(question)

    print("\nAnswer:")
    print(response)

    if response.source_nodes:
        top_node = response.source_nodes[0]
        file_name = top_node.node.metadata.get("file_name", "Unknown file")
        score = top_node.score
        text = top_node.node.get_content().replace("\n", " ")

        print("\nTop retrieved source:")
        print(f"Document name: {file_name}")
        print(f"Similarity score: {score}")
        print(f"Chunk preview: {text[:200]}")
    else:
        print("\nNo source nodes were retrieved.")


def print_all_sources(question, response):
    print("\n" + "=" * 80)
    print("Failure Test Question:")
    print(question)

    print("\nFull response:")
    print(response)

    print("\nAll retrieved source nodes:")
    for i, source_node in enumerate(response.source_nodes, start=1):
        file_name = source_node.node.metadata.get("file_name", "Unknown file")
        score = source_node.score
        text = source_node.node.get_content().replace("\n", " ")

        print(f"\nSource node {i}")
        print(f"Document name: {file_name}")
        print(f"Similarity score: {score}")
        print(f"Chunk preview: {text[:200]}")


# ------------------------------------------------------------
# Step 4: Query the Assistant
# ------------------------------------------------------------

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)
    print_top_source(question, response)

"""
Reflection on Step 4:
The assistant sounded most accurate when the retrieved document clearly matched
the question. For example, weekend hours should come from hours.txt, dairy-free
milk options should come from menu.txt, loyalty information should come from
loyalty.txt, and catering information should come from catering.txt.

This shows why RAG is useful: instead of guessing, the model can answer from
retrieved documents. I still need to check the source preview because an AI
answer can sound confident even when the retrieved text is weak.
"""


# ------------------------------------------------------------
# Step 5: Find a Failure
# ------------------------------------------------------------

failure_question = (
    "What is the owner's favorite coffee drink and how much profit did the shop make last year?"
)

failure_response = query_engine.query(failure_question)
print_all_sources(failure_question, failure_response)

"""
Failure reflection:
I asked this because the owner's favorite coffee drink and the shop's yearly
profit are probably not in the documents. I expected the assistant to struggle.

If retrieval fails, it may still retrieve coffee-related documents, but those
documents do not actually answer the question. If the model guesses anyway, that
is a hallucination risk. If the model says the information is not available in
the documents, that is a safer response.

To improve the system, I would add a stronger instruction telling the model:
"If the answer is not in the provided documents, say that you do not know based
on the documents." I would also add more documents if the business wants the
assistant to answer questions about owner details or financial information.
"""


# ------------------------------------------------------------
# Step 6: Reflection
# ------------------------------------------------------------

"""
Final Reflection:

1. The manual semantic RAG lesson required many steps: chunking, embedding,
storing vectors, searching, and building prompts. In this project, LlamaIndex
handled the main RAG process in only a few important lines:
- SimpleDirectoryReader loads documents
- VectorStoreIndex.from_documents builds the index
- index.as_query_engine creates the query system
This shows that a framework saves time and reduces repeated code.

2. A different business use case:
A hotel could use RAG to answer staff questions from SOP documents. For example,
front desk employees could ask about late checkout policy, no-show handling,
room assignment rules, or refund procedures. This would help employees find
accurate information faster.

3. One failure mode RAG cannot fully prevent:
RAG cannot fully prevent hallucination. Even with good retrieval, the model can
misunderstand the document, combine details incorrectly, or sound confident when
the answer is not really supported.
"""