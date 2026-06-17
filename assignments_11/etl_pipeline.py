# Video link: PASTE_YOUR_VIDEO_LINK_HERE_AFTER_RECORDING
#https://drive.google.com/file/d/1EOt3afSps7vdnxlCPLeLPYVi3RoFCt6j/view?usp=sharing
#https://drive.google.com/file/d/1UmIwL6P7OQvM6fNPrPq1iyIFg84pXGM3/view?usp=sharing
#https://drive.google.com/file/d/1Q83BDODXXbO--IV_yZFO_m5w5E66TGSC/view?usp=sharing

"""
Week 11 Cloud ETL Capstone

This pipeline:
1. Extracts hourly weather data from Open-Meteo
2. Transforms the first 24 hourly records using OpenAI classification
3. Loads the enriched JSON result to Azure Blob Storage
4. Runs as a Prefect flow
"""

import json
import os
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


load_dotenv()


CITY_NAME = "Boston, Massachusetts"
LATITUDE = 42.3601
LONGITUDE = -71.0589

OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are classifying hourly weather conditions for outdoor running.
Given a temperature in Celsius and a precipitation amount in mm,
classify the conditions as exactly one of: good, marginal, or bad.
Reply with that one word only -- no punctuation, no explanation.
""".strip()


@task(retries=2, retry_delay_seconds=10)
def extract_weather_data() -> dict[str, Any]:
    """
    Extract 7 days of hourly weather data from Open-Meteo.
    Returns raw JSON as a Python dictionary.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,precipitation",
        "forecast_days": 7,
        "timezone": "America/New_York",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    print(f"Extract completed for {CITY_NAME}.")
    print(f"Received hourly fields: {list(data.get('hourly', {}).keys())}")

    return data


@task
def transform_weather_data(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Reshape Open-Meteo hourly parallel lists into per-hour records.
    Classify the first 24 records using the OpenAI API.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    client = OpenAI(api_key=api_key)

    hourly = raw_data.get("hourly", {})

    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    precipitations = hourly.get("precipitation", [])

    if not times or not temperatures or not precipitations:
        raise ValueError("Missing hourly weather data from API response.")

    records = []

    for index, timestamp in enumerate(times[:24]):
        temperature_c = temperatures[index]
        precipitation_mm = precipitations[index]

        user_prompt = (
            f"Temperature: {temperature_c} Celsius\n"
            f"Precipitation: {precipitation_mm} mm"
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        classification = response.choices[0].message.content.strip().lower()

        if classification not in {"good", "marginal", "bad"}:
            classification = "unknown"

        record = {
            "city": CITY_NAME,
            "time": timestamp,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
            "running_condition": classification,
        }

        records.append(record)

        if (index + 1) % 6 == 0:
            print(f"Transform progress: classified {index + 1} records.")

    print(f"Transform completed. Enriched {len(records)} records.")

    return records


@task
def load_weather_data(records: list[dict[str, Any]]) -> str:
    """
    Load enriched records to Azure Blob Storage as JSON.
    Blob path: final/<today>/weather_etl.json
    """

    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pipeline-data")

    if not storage_account_name:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_NAME is missing. Add it to your .env file."
        )

    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    account_url = f"https://{storage_account_name}.blob.core.windows.net"

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential,
    )

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_path,
    )

    json_bytes = json.dumps(records, indent=2).encode("utf-8")

    blob_client.upload_blob(json_bytes, overwrite=True)

    print(f"Load completed.")
    print(f"Uploaded to: {container_name}/{blob_path}")
    print(f"Byte count: {len(json_bytes)}")

    return blob_path


@flow(log_prints=True)
def weather_etl_flow() -> str:
    """
    Full ETL flow:
    Extract -> Transform -> Load
    """

    raw_data = extract_weather_data()
    enriched_records = transform_weather_data(raw_data)
    final_blob_path = load_weather_data(enriched_records)

    print(f"Pipeline completed successfully. Final blob path: {final_blob_path}")

    return final_blob_path


if __name__ == "__main__":
    weather_etl_flow()