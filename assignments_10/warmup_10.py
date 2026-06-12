# Week 10 Warmup
# LLMs in Pipelines

# --- LLMs as Transform ---
# Q1

# 1. Parse the string "Jan 5th, 2024" into ISO date format.
# I would use deterministic code because date parsing follows predictable rules and does not require understanding meaning.

# 2. Classify a customer support ticket: "my card was charged twice".
# I would use an LLM because the task requires understanding natural language and classifying meaning into billing, technical, or general.

# 3. Calculate the average of a list of numbers.
# I would use deterministic code because math should be exact, fast, and reliable.

# 4. Extract company name from "Sr. Data Eng @ Acme Corp (contract)".
# I would use an LLM if the text format is messy and inconsistent, because it can understand freeform patterns.

# 5. Determine whether a product review is more than 100 words long.
# I would use deterministic code because counting words is simple, exact, and cheaper than an LLM.


def parse_date_example():
    from datetime import datetime

    text = "Jan 5th, 2024"
    cleaned = text.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
    date_obj = datetime.strptime(cleaned, "%b %d, %Y")
    return date_obj.strftime("%Y-%m-%d")


def average_numbers(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def is_review_more_than_100_words(review):
    return len(review.split()) > 100


# Q2
# Problem:
# The prompt "Summarize this product review in a few sentences" is too open-ended for a pipeline.
# The model might return different formats every time, which makes the output hard to parse and store in a database.
# A pipeline needs consistent, predictable, structured output.
#
# Better prompt:
# You are summarizing product reviews for a data pipeline.
# Return valid JSON only with these exact keys:
# {
#   "summary": "one sentence summary",
#   "sentiment": "positive, neutral, or negative",
#   "main_issue": "short phrase or null"
# }


# Q3
# If 50,000 records each take 1 second sequentially:
# 50,000 seconds / 60 = 833.33 minutes
# 833.33 minutes / 60 = about 13.9 hours.
#
# One practical strategy is batch or parallel processing.
# For example, process multiple records at the same time using async requests or worker batches,
# while respecting rate limits and retry rules.


# --- Azure OpenAI ---
# Q1
# Two reasons an organization might use Azure OpenAI:
# 1. Enterprise security and compliance: Azure OpenAI can fit into an organization's Azure security, identity, and compliance systems.
# 2. Production governance: companies can manage deployments, regions, access control, monitoring, and networking inside Azure.


# Q2
# Azure-specific client parameters:
# 1. azure_endpoint: the URL of the Azure OpenAI resource.
# 2. api_version: the Azure OpenAI API version to use.
# 3. azure_deployment: the deployment name created in Azure AI Foundry / Azure OpenAI Studio.
#
# Note: api_key is still needed, but the question asks for Azure-specific parameters, not the standard api_key.


# Q3
# When using AzureOpenAI, the model parameter does not take a normal model name like "gpt-4o-mini".
# It takes the Azure deployment name.
# You find this value in Azure AI Foundry or Azure OpenAI Studio under your model deployments.