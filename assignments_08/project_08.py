# project_08.py
# Run this in Azure Cloud Shell after completing the Cost Analysis above.

# Hourly rates from the Azure Pricing Calculator estimates.
rate_a = 0.010375   # Standard_B1s hourly rate for Scenario A
rate_b = 3.06       # Standard_NC6s_v3 hourly rate for Scenario B, VM only

hours_a = 160   # Scenario A: 8h/day, 5 days/week, about 4 weeks
hours_b = 730   # Scenario B: always on for full month

cost_a = rate_a * hours_a
cost_b_gpu = rate_b * hours_b

# Additional Scenario B services from Azure Pricing Calculator.
sql_database_cost = 741.16
blob_storage_cost = 1.66

scenario_b_total = cost_b_gpu + sql_database_cost + blob_storage_cost

print("=== Monthly Cost Estimates ===")
print(f"Scenario A (lightweight):       ${cost_a:.2f}")
print(f"Scenario B (GPU VM only):       ${cost_b_gpu:.2f}")
print(f"Scenario B SQL Database:        ${sql_database_cost:.2f}")
print(f"Scenario B Blob Storage:        ${blob_storage_cost:.2f}")
print(f"Scenario B total estimate:      ${scenario_b_total:.2f}")

if cost_a > 0:
    print(f"Scenario B total costs {scenario_b_total / cost_a:.1f}x more than Scenario A")
else:
    print("Please update rate_a with the hourly rate from Azure Pricing Calculator.")