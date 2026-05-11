from dotenv import load_dotenv
from openai import OpenAI
import json

# -----------------------------
# Task 1: Setup and System Prompt
# -----------------------------

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"

YOUR_SYSTEM_PROMPT = """
You are a job application coach helping career changers improve their job application materials.

Stay focused on resumes, cover letters, job application questions, and interview preparation.
Use clear, professional, practical language.

Do not invent experience, degrees, certifications, numbers, or results that the user did not provide.
Always remind the user to review and edit the output before submitting it anywhere.
Acknowledge that you may not know every specific industry norm, so the user should use their own judgment.
"""

# Deliberate choice:
# I told the assistant not to invent facts because job application materials must be honest.
# I also included a reminder to review and edit output because AI-generated content may need correction.


def get_completion(messages, model=MODEL, temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=400,
    )
    return response.choices[0].message.content


# -----------------------------
# Task 2: Bullet Point Rewriter
# -----------------------------

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
Use strong action verbs.
Do not invent facts that are not implied by the original bullet.

Return ONLY a valid JSON list.
Each item must have exactly these two keys:
"original" and "improved"

Do not wrap the JSON in markdown.
Do not use ```json.
Do not include any explanation before or after the JSON.

Bullet points:
```{bullet_text}```
"""

    messages = [{"role": "user", "content": prompt}]
    response_text = get_completion(messages, temperature=0.3)

    clean_response = response_text.strip()

    if clean_response.startswith("```json"):
        clean_response = clean_response.replace("```json", "", 1).strip()

    if clean_response.startswith("```"):
        clean_response = clean_response.replace("```", "", 1).strip()

    if clean_response.endswith("```"):
        clean_response = clean_response[:-3].strip()

    try:
        results = json.loads(clean_response)
    except json.JSONDecodeError:
        print("The model did not return valid JSON. Raw response:")
        print(response_text)
        return []

    print("\nRewritten Resume Bullets")
    print("-" * 50)

    for item in results:
        print("\nOriginal:")
        print(item["original"])
        print("Improved:")
        print(item["improved"])

    return results


# Test bullets:
# These bullets are weak because they are general and do not show specific action,
# measurable results, or clear impact. The model should improve them by using stronger
# verbs and making the value of the work clearer.
test_bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time",
]


# -----------------------------
# Task 3: Cover Letter Generator
# -----------------------------

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
Do not invent facts that are not provided.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information — which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)


# Few-shot reflection:
# I chose examples that connect the person's past experience to the new role.
# Few-shot prompting helps control the tone, structure, and specificity of the output.


# -----------------------------
# Task 4: Moderation Check
# -----------------------------

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    flagged = result.results[0].flagged

    if flagged:
        print(
            "Job Application Helper: I cannot help with that wording. Please rephrase it in a safe and respectful way."
        )
        return False

    return True


def run_tests():
    print("\n=== Testing Bullet Rewriter ===")
    rewrite_bullets(test_bullets)

    print("\n=== Testing Cover Letter Generator ===")
    paragraph = generate_cover_letter(
        "Junior Data Engineer",
        "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas.",
    )
    print(paragraph)

    print("\n=== Testing Moderation ===")
    safe_text = "Can you help me improve my resume?"
    unsafe_text = "I want to hurt someone."

    print("Safe test:", is_safe(safe_text))
    print("Flagged test:", is_safe(unsafe_text))


# -----------------------------
# Task 5: Chatbot Loop
# -----------------------------

def run_chatbot():
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            if raw_bullets:
                rewrite_bullets(raw_bullets)
                print("\nReminder: Please review and edit these bullets before submitting them anywhere.")
            else:
                print("No bullet points were provided.")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if not is_safe(job_title) or not is_safe(background):
                continue

            paragraph = generate_cover_letter(job_title, background)

            print("\nCover Letter Opening:")
            print(paragraph)
            print("\nReminder: Please review and edit this before submitting it anywhere.")

        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nJob Application Helper:")
            print(reply)

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_tests()
    run_chatbot()


"""
Task 6: Ethics Reflection
Option A — Comment block

A job application bot can produce biased advice because it was trained on text written by and about
many different groups, but not all groups are represented equally. It might favor corporate language,
certain industries, or communication styles that do not fit every person's background or culture.

If a job seeker submitted the bot's output directly without reviewing it, the application might include
incorrect, exaggerated, or generic information. This could hurt the applicant's chances or make the
application sound less authentic.

One guardrail I would add professionally is a clear reminder before the user copies the output:
"Review carefully before submitting. Make sure everything is accurate and personally true."
I would also keep a moderation filter and prevent the bot from inventing experience or qualifications.
"""