# Lesson 8 Warmup - Cloud Intro

## Cloud Concepts

### Cloud Concepts Question 1

Cloud computing means using computing resources like servers, storage, databases, and software over the internet instead of only using my own local computer. For example, I can use Azure to store files, run applications, or create virtual machines without buying physical hardware.

### Cloud Concepts Question 2

The cloud is useful because it helps people and companies save money, access resources from anywhere, and scale up or down when needed. Instead of buying expensive servers, a business can pay only for the resources it uses.

### Cloud Concepts Question 3

A virtual machine is a computer that runs in the cloud instead of on my physical laptop. It has an operating system, CPU, memory, and storage, but it is managed through a cloud provider like Azure.

## Cloud Concepts Scenarios

### Scenario 1

If a small business needs a website, using the cloud is helpful because the business does not need to buy and maintain its own server. Azure can host the website and allow the business to increase resources if more customers visit the site.

### Scenario 2

If a company has a lot of data to store, cloud storage is a good solution because it can store files safely and allow access from different locations. It also helps with backup and disaster recovery.

### Scenario 3

If an application suddenly gets more users, cloud computing helps because the company can scale the resources up. When fewer users are using the application, the company can scale down to save money.

### Cloud Concepts Question 4

Scalability means the ability to increase or decrease computing resources based on demand. For example, if more people visit a website, the cloud can add more resources so the website does not slow down.

### Cloud Concepts Question 5

Pay-as-you-go pricing means I only pay for the cloud resources I actually use. For example, if I run a virtual machine for 160 hours, I pay for those 160 hours instead of buying a full physical server.

## Azure Basics

### Azure Basics Question 1

Microsoft Azure is Microsoft’s cloud computing platform. It provides services like virtual machines, storage, databases, networking, artificial intelligence tools, and application hosting.

### Azure Basics Question 2

An Azure Resource Group is a container that holds related Azure resources. For example, a virtual machine, storage account, and network can be placed in the same resource group so they are easier to organize and manage.

### Azure Basics Question 3

Azure Cloud Shell is a command-line tool inside the Azure Portal. It lets me run Azure CLI commands from the browser without installing Azure tools on my own computer.

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