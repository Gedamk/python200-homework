from dotenv import load_dotenv
from openai import OpenAI
import json

# -----------------------------
# Setup
# -----------------------------

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"


# -----------------------------
# The Chat Completions API
# -----------------------------

print("\n=== API Question 1 ===")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

print("Response text:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal tokens used:")
print(response.usage.total_tokens)


print("\n=== API Question 2 ===")

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    print(f"\nTemperature {temperature}:")
    print(response.choices[0].message.content)

# Observation:
# Temperature 0 gives more predictable and consistent output.
# Higher temperatures like 0.7 and 1.5 create more varied and creative responses.
# If I needed consistent, reproducible output, I would use temperature=0.


print("\n=== API Question 3 ===")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

for index, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {index}:")
    print(choice.message.content)


print("\n=== API Question 4 ===")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work.",
        }
    ],
    max_tokens=15,
)

print(response.choices[0].message.content)

# Observation:
# The response was cut short because max_tokens limits how many tokens the model can generate.
# In a real application, max_tokens helps control cost, speed, and response length.


# -----------------------------
# System Messages and Personas
# -----------------------------

print("\n=== System Question 1 ===")

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is.",
    },
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)

print("\nPatient tutor response:")
print(response.choices[0].message.content)

messages = [
    {
        "role": "system",
        "content": "You are a direct senior software engineer. Explain clearly and professionally, with no extra encouragement.",
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is.",
    },
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)

print("\nSenior engineer response:")
print(response.choices[0].message.content)

# Observation:
# Changing the system message changed the model's tone and personality.
# The first response was more encouraging, while the second was more direct.


print("\n=== System Question 2 ===")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)

print(response.choices[0].message.content)

# Explanation:
# The model knows Jordan's name because I passed the earlier conversation history
# in the messages list. The API is stateless, so it only knows what I send each time.


# -----------------------------
# Prompt Engineering
# -----------------------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]


print("\n=== Prompt Question 1: Zero-Shot ===")

prompt = f"""
Classify each review as positive, negative, or mixed.

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)


print("\n=== Prompt Question 2: One-Shot ===")

prompt = f"""
Classify each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)

# Observation:
# Adding one example helps the model understand the exact format I want.


print("\n=== Prompt Question 3: Few-Shot ===")

prompt = f"""
Classify each review as positive, negative, or mixed.

Examples:
Review: "The onboarding was easy and the team was supportive."
Sentiment: positive

Review: "The app crashes every day and support is not helpful."
Sentiment: negative

Review: "The price is good, but the setup instructions are confusing."
Sentiment: mixed

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)

# Comparison:
# Zero-shot is useful for simple tasks.
# One-shot is useful when I want to show the model the desired format.
# Few-shot is useful when I want more consistent results across several examples.


print("\n=== Prompt Question 4: Chain of Thought ===")

prompt = """
Solve this problem. Show the calculation step by step, then give a clearly labeled final answer.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)

# Explanation:
# Asking the model to reason step by step can improve accuracy because it breaks
# the problem into smaller parts before giving the final answer.


print("\n=== Prompt Question 5: Structured Output JSON ===")

review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

prompt = f"""
Analyze the review below.

Return ONLY raw valid JSON with these keys:
sentiment, confidence, reason

Do not wrap the JSON in markdown.
Do not use ```json.
Do not include any explanation before or after the JSON.

Review:
```{review}```
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

raw_response = response.choices[0].message.content

print("\nRaw response:")
print(raw_response)

clean_response = raw_response.strip()

if clean_response.startswith("```json"):
    clean_response = clean_response.replace("```json", "", 1).strip()

if clean_response.startswith("```"):
    clean_response = clean_response.replace("```", "", 1).strip()

if clean_response.endswith("```"):
    clean_response = clean_response[:-3].strip()

try:
    parsed = json.loads(clean_response)
    print("\nParsed JSON:")
    print("Sentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])
except json.JSONDecodeError:
    print("\nThe response was not valid JSON. Raw response for debugging:")
    print(raw_response)


print("\n=== Prompt Question 6: Delimiters ===")

user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print("\nInstruction text result:")
print(response.choices[0].message.content)

regular_text = "Python is a popular programming language used in data science, web development, and automation."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{regular_text}```
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print("\nRegular text result:")
print(response.choices[0].message.content)

# Explanation:
# Delimiters help separate user content from instructions.
# This prevents the model from confusing the user's text with the command.


# -----------------------------
# Local Models with Ollama
# -----------------------------

print("\n=== Ollama Question 1 ===")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences.",
        }
    ],
)

print("\nOpenAI response:")
print(response.choices[0].message.content)

"""
Paste your Ollama output here after running this command in terminal:

ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

Reflection:
The OpenAI response was more polished and clear.
One advantage of running a local model is privacy and not needing an API call.
One disadvantage is that a small local model may be less accurate or less detailed.
"""