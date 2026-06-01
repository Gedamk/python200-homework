# Lesson 9 Warmup - Data in the Cloud

# --- Azure Authentication ---

# Q1
# When I run a Python script locally that uses DefaultAzureCredential,
# it relies on my local Azure login session. Before running the script,
# I need to run `az login` in the terminal. After I log in, Azure CLI stores
# my login session. DefaultAzureCredential checks available credential sources,
# finds the Azure CLI login, and uses it to authenticate my Python script.

# Q2
# A deployed pipeline running on an Azure VM or container cannot use `az login`
# because there is no human there to open a browser and sign in. Instead,
# deployed Azure resources should use Managed Identity. Managed Identity gives
# the VM or container its own Azure identity. The same Python code works without
# changes because DefaultAzureCredential can use Azure CLI credentials locally
# and Managed Identity when running in Azure.

# Q3
# If I get an AuthenticationError immediately, two likely causes are:
#
# 1. I am not logged in to Azure CLI.
#    To diagnose this, I would run `az account show`. If it fails, I would run
#    `az login` again.
#
# 2. I am logged in but my account does not have permission to access the
#    storage account or resource.
#    To diagnose this, I would check the Azure Portal permissions/IAM settings
#    for the storage account and confirm that my account has the correct role,
#    such as Storage Blob Data Contributor.


# --- Blob Storage ---

# Q1
# Azure Blob Storage has a three-level hierarchy:
#
# 1. Storage Account: the top-level Azure storage service.
# 2. Container: a folder-like grouping inside the storage account.
# 3. Blob: the actual file stored inside the container.
#
# Analogy:
# A storage account is like a filing cabinet.
# A container is like a drawer in the filing cabinet.
# A blob is like a file inside the drawer.

# Q2
# Scenario 1:
# I would use Blob Storage because raw JSON API responses are files that can
# be saved and reprocessed later.
#
# Scenario 2:
# I would use a relational database like Azure SQL because 50 million customer
# transactions need to be queried by date range and customer ID.
#
# Scenario 3:
# I would use Blob Storage because NumPy arrays or image embeddings are file-like
# model outputs that need to be saved between pipeline runs.


# Q3
def list_container(container_client):
    """
    Print the name and size in bytes of every blob in a container.
    """
    for blob in container_client.list_blobs():
        print(f"{blob.name} - {blob.size} bytes")


# Q4
def upload_text(container_client, blob_name, text):
    """
    Encode a Python string as UTF-8 and upload it as a blob.
    If the blob already exists, overwrite it.
    """
    data = text.encode("utf-8")
    container_client.upload_blob(name=blob_name, data=data, overwrite=True)