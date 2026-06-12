"""
Week 10 Project: LLM Transform Pipeline

Video link:
PASTE_YOUR_VIDEO_LINK_HERE

Reflection:
Classifying weather conditions for outdoor running can be done with an LLM, but it may not be the best use of an LLM.
Because the input only has temperature and precipitation, deterministic rules could probably do the job faster, cheaper, and more consistently.
For example, a rule like "temperature > 10 and precipitation < 1 means good" would be easier to test.
The LLM approach is useful for practicing how language models can work inside a pipeline, but a rule-based approach would be better for this simple numeric dataset.
"""

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# Fill these in using your own Azure values.
ACCOUNT_URL = os.getenv(
    "AZURE_STORAGE_ACCOUNT_URL",
    "https://<your-account>.blob.core.windows.net",
)

CONTAINER = "pipeline-data"

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


def reshape_weather_data(weather_json):
    """
    Convert the hourly parallel lists into a list of hourly records.

    Example input:
    {
        "hourly": {
            "time": [...],
            "temperature_2m": [...],
            "precipitation": [...]
        }
    }

    Example output:
    [
        {
            "time": "2026-06-09T00:00",
            "temperature_2m": 18.5,
            "precipitation": 0.0
        }
    ]
    """
    hourly = weather_json["hourly"]

    records = []
    for time_value, temp_value, precip_value in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"],
    ):
        records.append(
            {
                "time": time_value,
                "temperature_2m": temp_value,
                "precipitation": precip_value,
            }
        )

    return records


def load_fallback_weather_data():
    """
    Load fallback data if Blob Storage raw file is not available.
    """
    fallback_path = Path("assignments/resources/weather_raw.json")

    if not fallback_path.exists():
        raise FileNotFoundError(
            "Fallback file not found. Expected assignments/resources/weather_raw.json"
        )

    with fallback_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_blob_service_client():
    """
    Connect to Azure Blob Storage using your active az login session.
    """
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)


def download_raw_weather(blob_service_client, upload_date):
    """
    Download raw weather data from:
    raw/<date>/weather.json
    """
    blob_path = f"raw/{upload_date}/weather.json"

    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER,
        blob=blob_path,
    )

    downloaded = blob_client.download_blob().readall()
    return json.loads(downloaded)


def classify_conditions(client, record):
    """
    Use OpenAI to classify one hourly weather record.
    """
    user_message = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in VALID_LABELS:
        return "unknown"

    return label


def upload_processed_weather(blob_service_client, upload_date, enriched_records):
    """
    Upload enriched records to:
    processed/<date>/weather_classified.json
    """
    output_blob_path = f"processed/{upload_date}/weather_classified.json"

    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER,
        blob=output_blob_path,
    )

    blob_client.upload_blob(
        json.dumps(enriched_records, indent=2),
        overwrite=True,
    )

    print(f"Uploaded processed file to: {output_blob_path}")


def download_processed_weather(blob_service_client, upload_date):
    """
    Download the processed file for spot-checking.
    """
    processed_blob_path = f"processed/{upload_date}/weather_classified.json"

    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER,
        blob=processed_blob_path,
    )

    downloaded = blob_client.download_blob().readall()
    return json.loads(downloaded)


def save_first_10_records(enriched_records):
    """
    Save first 10 records to outputs/first_10_records.json
    """
    output_dir = Path("assignments_10/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "first_10_records.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(enriched_records[:10], file, indent=2)

    print(f"Saved first 10 records to: {output_path}")


def main():
    upload_date = os.getenv("WEATHER_DATA_DATE", date.today().isoformat())

    print(f"Using weather data date: {upload_date}")
    print("Connecting to Azure Blob Storage...")

    blob_service_client = get_blob_service_client()

    try:
        print("Downloading raw weather data from Blob Storage...")
        weather_json = download_raw_weather(blob_service_client, upload_date)
    except Exception as error:
        print("Could not download raw data from Blob Storage.")
        print(f"Reason: {error}")
        print("Trying fallback dataset...")
        weather_json = load_fallback_weather_data()

    records = reshape_weather_data(weather_json)
    records = records[:24]

    print(f"Processing {len(records)} records...")

    openai_client = OpenAI()

    enriched_records = []

    for index, record in enumerate(records, start=1):
        conditions = classify_conditions(openai_client, record)

        enriched_record = {
            **record,
            "conditions": conditions,
        }

        enriched_records.append(enriched_record)

        if index % 6 == 0:
            print(f"Processed {index} records...")

    upload_processed_weather(blob_service_client, upload_date, enriched_records)

    processed_records = download_processed_weather(blob_service_client, upload_date)

    df = pd.DataFrame(processed_records)

    print("\nCondition counts:")
    print(df["conditions"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

    save_first_10_records(enriched_records)

    print("\nDone.")


if __name__ == "__main__":
    main()