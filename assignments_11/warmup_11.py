"""
Week 11 Warmup
Cloud ETL Capstone
"""

from prefect import task, get_run_logger


# ------------------------------------------------------------
# Prefect Orchestration
# ------------------------------------------------------------

# Prefect Question 1
"""
A @task is one step inside a pipeline. It usually does one specific piece of work,
such as calling an API, transforming data, or uploading a file.

A @flow is the full workflow that controls the order of tasks. The flow connects
tasks together and lets Prefect monitor the whole pipeline.

If I have a helper function that converts Celsius to Fahrenheit, I would usually
not decorate it with @task because it is a simple in-memory calculation with no
I/O. Keeping it as a normal helper function makes the code simpler. I would use
@task when I want Prefect to track, retry, or log that step separately.
"""


# Prefect Question 2
# Decorator line only:
# @task(retries=3, retry_delay_seconds=30)


# Prefect Question 3
"""
If the Prefect UI shows extract Completed, transform Failed, and load never ran,
I would open the failed flow run in the Prefect UI. Then I would click the
transform task to inspect its task run details and logs.

I would expect to find the Python error message, traceback, task logs, input
information, and the exact point where the transform task failed. Since transform
failed, the load task never ran because downstream tasks depend on successful
upstream tasks.
"""


# ------------------------------------------------------------
# Production Patterns
# ------------------------------------------------------------

# Production Question 1
"""
raise_for_status() checks the HTTP response from an API call. If the API returns
an error status code such as 400 or 500, it raises an exception.

This is better than only writing:

if response.status_code != 200:
    print("error")

because printing an error does not always stop the pipeline. If the task only
prints an error, downstream tasks might still run with bad or missing data.

With raise_for_status(), Prefect sees the task as Failed, records the error,
and prevents downstream tasks from running. If the API returns a 500 error,
the extract task fails clearly and the transform/load tasks do not run.
"""


# Production Question 2
"""
overwrite=True protects me when I re-run the pipeline and write to the same blob
path, such as final/{today}/weather_etl.json.

If the pipeline crashes halfway through the transform step, I can fix the bug and
run the pipeline again from the beginning. When the load step reaches Azure Blob
Storage, overwrite=True allows the new corrected result to replace the older file
at the same path.

Without overwrite=True, the upload could fail if that blob already exists, and I
would have to manually delete the old blob or create a new filename.
"""


# Production Question 3
@task
def load_records_stub(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")