# https://drive.google.com/file/d/1decZgkHyGOFiYPx66MB2XYO-W-NZuiyu/view?t=112.595
# https://drive.google.com/file/d/102HWpNir2pBweXqoU25ytzM-BXrTh8US/view?t=30.306

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


ACCOUNT_URL = "https://gedamctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"


def extract_weather_data():
    """
    Extract 7 days of hourly weather data from the Open-Meteo API.
    This example uses Charlotte, NC.
    """
    latitude = 35.2271
    longitude = -80.8431

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    return response.json()


def upload_weather_data(container_client, weather_data):
    """
    Serialize weather data to JSON bytes and upload it to Blob Storage.
    """
    today = date.today().isoformat()
    blob_name = f"raw/{today}/weather.json"

    json_bytes = json.dumps(weather_data, indent=2).encode("utf-8")

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(json_bytes, overwrite=True)

    print(f"Uploaded blob: {blob_name}")
    print(f"Bytes uploaded: {len(json_bytes)}")

    return blob_name


def list_blobs(container_client):
    """
    List all blobs in the container.
    """
    print("\nBlobs in container:")
    for blob in container_client.list_blobs():
        print(f"{blob.name} - {blob.size} bytes")


def read_back_weather_data(container_client, blob_name):
    """
    Download the uploaded blob, save it locally, and load hourly data into pandas.
    """
    blob_client = container_client.get_blob_client(blob_name)

    downloaded_bytes = blob_client.download_blob().readall()
    downloaded_text = downloaded_bytes.decode("utf-8")
    downloaded_data = json.loads(downloaded_text)

    output_dir = Path("assignments_09/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "weather_raw.json"
    output_file.write_text(downloaded_text, encoding="utf-8")

    print(f"\nSaved downloaded JSON to: {output_file}")

    hourly_data = downloaded_data["hourly"]
    df = pd.DataFrame(hourly_data)

    print("\nFirst 5 rows of hourly weather data:")
    print(df.head())

    return df


def main():
    credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )

    container_client = blob_service_client.get_container_client(CONTAINER)

    weather_data = extract_weather_data()
    blob_name = upload_weather_data(container_client, weather_data)
    list_blobs(container_client)
    read_back_weather_data(container_client, blob_name)


if __name__ == "__main__":
    main()