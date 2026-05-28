# Week 8 Project: Azure Intro and Cost Analysis

## Video Link

Paste my video link here:

[Video Link](PASTE-YOUR-VIDEO-LINK-HERE)

## Portal Walkthrough Summary

In my video, I showed the Azure Portal while logged in under the Code the Dream tenant. I navigated to my personal resource group and pointed out the storage account inside it.

I also opened Azure Cloud Shell and ran commands to check my environment. I used `ls ~/clouddrive` to show that my persistent storage was working, and I used `ls ~/.ssh` to show that my SSH private and public key files were present.

I also ran:

```bash
az group list --output table
```

This command showed my Azure resource group:

```text
Name                Location    Status
------------------  ----------  ---------
p200-2026-gedam-rg  eastus      Succeeded
```

## Scenario A — Lightweight Compute

Service: Standard_B1s VM
Region: East US
Operating System: Linux
Usage: 160 hours per month

Estimated hourly rate: $0.10
Estimated monthly cost: $16.00 per month

This scenario is cheaper because it uses a small virtual machine and only runs part time.

## Scenario B — Heavy Analytics Workload

Services:

* Standard_NC6s_v3 GPU VM
* Azure SQL Database, General Purpose, 4 vCores
* Azure Blob Storage, 1 TB

Region: East US
Operating System: Linux
Usage: 730 hours per month for the GPU VM

Estimated GPU VM hourly rate: $3.06
Estimated GPU VM monthly cost: $2,233.80

Estimated total cost: $____ per month

This scenario is much more expensive because the GPU VM runs all month and the full setup also includes SQL Database and Blob Storage.

## What I Found Interesting

I found that cloud costs can change a lot depending on the size of the virtual machine, how many hours it runs, and whether extra services like SQL Database and Blob Storage are included. The GPU VM is much more expensive than the small lightweight VM.

## Python Script Output

```text
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $16.00
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 139.6x more than Scenario A
```

The Python script matched the VM cost estimates from the hourly rates I used. Scenario B’s full total may be higher because the complete scenario also includes SQL Database and Blob Storage.
