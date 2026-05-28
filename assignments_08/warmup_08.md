# Week 8 Warmup

## Cloud Concepts Question 1

The core economic model of cloud computing is pay-as-you-go. This means a company pays for computing resources only when they use them, instead of buying and maintaining their own physical servers.

Owning your own servers means you pay a large cost upfront, and you are responsible for maintenance, upgrades, security, electricity, and replacement. Cloud computing is more flexible because you can increase or decrease resources when needed.

## Cloud Concepts Question 2

Vertical scaling means making one machine more powerful. For example, if a data scientist needs faster model training, they may choose a machine with more RAM, more CPU, or a better GPU.

Horizontal scaling means adding more machines to share the work. For example, if a web application gets many more users, the company can add more servers to handle the traffic.

Scenario 1: A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

This is horizontal scaling because the application needs more servers to handle many more users at the same time.

Scenario 2: A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

This is vertical scaling because one machine is being upgraded to become more powerful.

Scenario 3: A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

This is horizontal scaling because the file processing work can be divided across many machines.

## Cloud Concepts Question 3

Gmail: SaaS because users just use the finished email software and do not manage the servers or application infrastructure.

Azure Virtual Machines: IaaS because Azure provides the virtual computer, but the user manages the operating system, software, updates, and applications.

Azure App Service: PaaS because developers deploy their application code, while Azure manages most of the server, runtime, and scaling.

AWS S3: PaaS because it is a managed storage service. The user stores files and objects, but AWS manages the storage infrastructure.

GitHub Codespaces: PaaS because it gives developers a ready-to-use coding environment in the cloud without managing the underlying machine directly.

Snowflake: SaaS because it is a managed data platform where users work with data, queries, and warehouses without managing the underlying infrastructure.

IaaS means Infrastructure as a Service. It gives the user basic cloud infrastructure such as virtual machines, storage, and networking. Example: Azure Virtual Machines. As the developer, I am responsible for managing the operating system, installed software, updates, security settings, and my application.

PaaS means Platform as a Service. It gives the developer a platform to deploy applications without managing the full server. Example: Azure App Service. As the developer, I mainly manage my application code and configuration.

SaaS means Software as a Service. It is finished software that users access through the internet. Example: Gmail. As the user, I mainly manage my account, settings, and data.

## Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake is a cloud-based platform designed to help teams store, process, analyze, and manage large amounts of data.

It differs from using Azure directly because the platform handles many data engineering details for you. You gain easier setup, built-in data tools, scalability, and less infrastructure management. You give up some control because you are working inside the platform's rules, pricing model, and features.

## Cloud Concepts Question 5

The cloud may not be the right choice when the cost is too high for the amount of usage, especially if resources are running all the time and not managed carefully.

The cloud may also not be the right choice when there are strict privacy, security, or legal requirements that make it better to keep systems on local private infrastructure.

## Azure Basics Question 1

An Azure subscription is like the billing and access container for Azure services. It controls what resources can be created and how usage is charged.

A resource group is like a folder that organizes related Azure resources, such as storage accounts, virtual machines, and databases.

My personal resource group is mine for the course work. The subscription is shared or managed by Code the Dream for the class environment.

## Azure Basics Question 2

Azure Cloud Shell is ephemeral by default, which means the shell environment can reset and temporary files may disappear.

The course setup uses mounted storage, such as clouddrive, to make files persistent. This means files saved in the mounted storage can still be there after Cloud Shell restarts.

## Azure Basics Question 3

An SSH private key is the secret key that stays with me. I should not share it.

An SSH public key is the key that can be uploaded to remote systems. It is safe because the public key can verify my private key, but it cannot easily recreate the private key.

The public key gets uploaded to remote systems because it allows the remote system to recognize my computer securely when I connect.

## Azure Basics Question 4

Command:

```bash
az account show