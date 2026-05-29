### Azure Basics Question 4

I ran this command in Azure Cloud Shell:

```bash
az account show
```

Output:

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#gedam_ka@yahoo.com",
    "type": "user"
  }
}
```

When I run `az account show` without `--output table`, Azure shows detailed JSON output with many key-value pairs. When I add `--output table`, the output becomes shorter and easier to read in rows and columns.
