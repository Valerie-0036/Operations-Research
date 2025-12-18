import os
import redis
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Tuple

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.documents import Document

# NOTE: Ensure you have the Gurobi solver installed and a license configured.
# You also need the gurobipy package: pip install gurobipy

# 1. SETUP & CONFIGURATION
# ========================
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize Redis
nameofindex = "trial_skills_embedding"

config = RedisConfig(
    index_name=nameofindex,
    legacy_key_format=False,
    redis_url=REDIS_URL,
    distance_metric="L2"
)

vectorstore = RedisVectorStore(
    embeddings=embeddings,
    config=config
)

# 2. DATA INGESTION
# =================
employees = ['Emp_A', 'Emp_B', 'Emp_C', 'Emp_D', 'Emp_E'] # Set I

employee_skills = [
    ["Python", "SQL (PostgreSQL/MySQL)", "Cloud Infrastructure (AWS/Azure)", "Docker/Kubernetes", "System Architecture Design", "Agile/Scrum Mastery"],
    ["JavaScript (React/Vue)", "HTML5/CSS3 (Tailwind CSS)", "API Integration (REST/GraphQL)", "Unit Testing (Jest/Enzyme)", "Responsive Design", "User Empathy", "A/B Testing Analysis"],
    ["R/Pandas/NumPy", "Data Warehousing (Snowflake)", "ETL Pipeline Development", "Data Visualization (Tableau/Power BI)", "Critical Thinking", "Inference and Reporting"],
    ["Network Security (Firewalls, IDS/IPS)", "Penetration Testing (Kali Linux)", "Security Information and Event Management (SIEM)", "Shell Scripting (Bash/PowerShell)", "Incident Response", "Risk Assessment", "Pressure Management"],
    ["Terraform/Ansible", "Continuous Integration/Continuous Deployment (CI/CD)", "Linux Administration", "Monitoring Tools (Prometheus/Grafana)", "Version Control (Git)", "Time Management"]
]

# Adding documents to Redis with Employee Metadata
print("Ingesting data into Redis...")
# NOTE: The ingestion is commented out to prevent re-adding data on every run.
# for i, employee in enumerate(employees):
#     for skill in employee_skills[i]:
#         doc = Document(
#             page_content=skill,
#             metadata={
#                 "employee": employee,
#                 "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
#             }
#         )
#         vectorstore.add_documents([doc])

# 3. SEARCH & DISTANCE CALCULATION
# ================================
requirements = [
  "Python",
  "LangChain",
  "Penetration Testing",
  "Risk Management"
] #set J
print("Calculating semantic distances...")

top_k_data = {}
number_of_k = 5

for req in requirements:
    actual_embedding = embeddings.embed_query(req)
    results = vectorstore.similarity_search_with_score_by_vector(
        actual_embedding,
        k=number_of_k
    )
    
    candidates = []
    seen_employees = set()
    
    for doc, score in results:
        emp_name = doc.metadata.get("employee")
        if emp_name in employees and emp_name not in seen_employees:
            candidates.append((emp_name, score))
            seen_employees.add(emp_name)
            
    top_k_data[req] = candidates

# 4. OPTIMIZATION PRE-PROCESSING
# ==============================
valid_assignments = []
distance_map = {}

for task, candidates in top_k_data.items():
    for emp, dist in candidates:
        valid_assignments.append((emp, task))
        distance_map[(emp, task)] = dist

S = len(requirements) / len(employees)
alpha = 0.8
beta = (1.0 - alpha) * S

print(f"Optimization Parameters: Alpha={alpha}, Beta={beta:.3f}, S={S}")

# 5. GUROBI MODEL (SPARSE IMPLEMENTATION)
# ======================================
model = gp.Model("EmployeeAssignment")

# --- Variables ---
# p[i,j] = 1 if employee i is assigned to task j
p = model.addVars(valid_assignments, vtype=GRB.BINARY, name="assign")
# y[i] = 1 if employee i is used
y = model.addVars(employees, vtype=GRB.BINARY, name="active")

# --- Objective Function ---
# Goal 1: Quality (Sum of distances for chosen pairs)
quality = gp.quicksum(distance_map[i, j] * p[i, j] for i, j in valid_assignments)

# Goal 2: Efficiency (Sum of active employees)
headcount = y.sum()

model.setObjective((alpha * quality) + (beta * headcount), GRB.MINIMIZE)

# --- Constraints ---

# 1. Single Assignment Constraint
# Each requirement j must be assigned to EXACTLY one employee i
for j in requirements:
    model.addConstr(
        gp.quicksum(p[i, task] for i, task in valid_assignments if task == j) == 1,
        name=f"assign_one_{j}"
    )

# 2. Activation Constraint (Big-M)
# If employee i takes a task, y[i] must be 1
M = len(requirements)
for i in employees:
    model.addConstr(
        gp.quicksum(p[emp, j] for emp, j in valid_assignments if emp == i) <= M * y[i],
        name=f"activate_{i}"
    )

# 6. SOLVE AND OUTPUT
import time
# ===================
try:
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    wall_clock_time = end_time - start_time
    print("\n" + "="*30)
    print(" OPTIMIZATION RESULTS ")
    print("="*30)

    # Check if a feasible solution was found
    if model.Status == GRB.OPTIMAL or model.Status == GRB.FEASIBLE:
        total_quality_cost = sum(distance_map[i,j] * p[i,j].X for i,j in valid_assignments)
        active_count = sum(y[i].X for i in employees)

        print(f"Total Objective Score: {model.ObjVal:.4f}")
        print(f"Total Semantic Distance: {total_quality_cost:.4f}")
        print(f"Total Employees Used: {int(active_count)}")
        # --- ADDED TIMING OUTPUT ---
        print("-" * 30)
        print("PERFORMANCE METRICS:")
        # Access the Runtime attribute directly from the Gurobi model object
        print(f"Gurobi Solve Time: {model.Runtime:.4f} seconds")
        print(f"Wall Clock Time:      {wall_clock_time:.4f} seconds")
        print("-" * 30)
        for i in employees:
            if y[i].X > 0.5: # Check if employee is active
                print(f"\n🟢 Employee {i} (ACTIVE):")
                for j in requirements:
                    # Check if the assignment is valid and was selected
                    if (i, j) in p and p[i, j].X > 0.5:
                        dist = distance_map[(i,j)]
                        print(f"   ├── Assigned Task: {j}")
                        print(f"   └── Distance: {dist:.4f}")
    else:
        print("No optimal solution found.")
        print(f"Model Status Code: {model.Status}")

except gp.GurobiError as e:
    print(f"Gurobi Error Code {e.errno}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")