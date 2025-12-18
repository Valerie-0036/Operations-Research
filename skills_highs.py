import os
import redis
import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Tuple

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.documents import Document

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
# for i, employee in enumerate(employees):
#     for skill in employee_skills[i]:
#         doc = Document(
#             page_content=skill,
#             metadata={
#                 "employee": employee,
#                 "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
#             }
#         )
#         # Note: add_documents expects a list
#         vectorstore.add_documents([doc])

# 3. SEARCH & DISTANCE CALCULATION
# ================================
# requirements = [
#     "Cloud Security Posture Management (CSPM)",
#     "Identity and Access Management (IAM)",
#     "Kubernetes Security",
#     "Security as Code (SaC)",
#     "Network Segmentation (VPC/VNet)",
#     "Threat Modeling",
#     "Policy Development",
# ] # Set J
requirements = [
  "Python",
  "LangChain",
  "Penetration Testing",
  "Risk Management"
] #set J
print("Calculating semantic distances...")

# We need a dictionary: { 'Requirement_Name': [('Emp_A', 0.12), ('Emp_B', 0.15)] }
top_k_data = {}
number_of_k = 5

for req in requirements:
    actual_embedding = embeddings.embed_query(req)
    
    # CRITICAL CHANGE: We need the score (distance), not just the document.
    # Returns list of (Document, float_score)
    results = vectorstore.similarity_search_with_score_by_vector(
        actual_embedding, 
        k=number_of_k
    )
    
    candidates = []
    seen_employees = set() # To prevent adding the same employee twice if they match on 2 different skills
    
    for doc, score in results:
        emp_name = doc.metadata.get("employee")
        
        # If this employee is in our valid list and hasn't been added for this specific task yet
        if emp_name in employees and emp_name not in seen_employees:
            candidates.append((emp_name, score))
            seen_employees.add(emp_name)
            
    top_k_data[req] = candidates

# 4. OPTIMIZATION PRE-PROCESSING
# ==============================
# Flatten top_k_data into ValidPairs for the Sparse Matrix
valid_assignments = []
distance_map = {}

for task, candidates in top_k_data.items():
    for emp, dist in candidates:
        valid_assignments.append((emp, task))
        distance_map[(emp, task)] = dist

# Scaling Factor Calculation
# S balances the magnitude between "Total Distance" (approx 0.5 * 7 tasks = 3.5) 
# and "Headcount" (approx 5 employees).
# S = Max_Possible_Distance_Cost / Max_Headcount_Cost
S = len(requirements) / len(employees)

# Weights
alpha = 0.8  # Priority on Matching Quality
# Corrected Beta formula: ensures Beta is positive and scaled
beta = (1.0 - alpha) * S 

print(f"Optimization Parameters: Alpha={alpha}, Beta={beta:.3f}, S={S}")

# 5. PYOMO MODEL (SPARSE IMPLEMENTATION)
# ======================================
model = pyo.ConcreteModel()

# --- Sets ---
model.I = pyo.Set(initialize=employees)    # All Employees
model.J = pyo.Set(initialize=requirements) # All Tasks

# Sparse Set: Only allows assignments found in the Top K search
model.ValidPairs = pyo.Set(initialize=valid_assignments, dimen=2)

# --- Parameters ---
model.d = pyo.Param(model.ValidPairs, initialize=distance_map)
model.M = pyo.Param(initialize=len(requirements))

# --- Variables ---
model.p = pyo.Var(model.ValidPairs, domain=pyo.Binary) # Assignment
model.y = pyo.Var(model.I, domain=pyo.Binary)          # Employee Active?

# --- Objective Function ---
def objective_rule(model):
    # Goal 1: Quality (Sum of distances for chosen pairs)
    quality = sum(model.d[i, j] * model.p[i, j] for (i, j) in model.ValidPairs)
    
    # Goal 2: Efficiency (Sum of active employees)
    headcount = sum(model.y[i] for i in model.I)
    
    return (alpha * quality) + (beta * headcount)

model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# --- Constraints ---

# 1. Single Assignment Constraint
# Each requirement j must be assigned to EXACTLY one employee i
def one_employee_rule(model, j):
    # Filter: Only sum over employees who are actually in the ValidPairs list for this task
    candidates_for_j = [i for i in model.I if (i, j) in model.ValidPairs]
    
    # If no one in the top K can do this task, the model breaks (infeasible).
    # In production, you might add a 'dummy' employee with high cost to prevent crashing.
    if not candidates_for_j:
        return pyo.Constraint.Skip
        
    return sum(model.p[i, j] for i in candidates_for_j) == 1

model.AssignOne = pyo.Constraint(model.J, rule=one_employee_rule)

# 2. Activation Constraint (Big-M)
# If employee i takes a task, y[i] must be 1
def activation_rule(model, i):
    tasks_for_i = [j for j in model.J if (i, j) in model.ValidPairs]
    
    if not tasks_for_i:
        return model.y[i] == 0
        
    return sum(model.p[i, j] for j in tasks_for_i) <= model.M * model.y[i]

model.ActivateEmp = pyo.Constraint(model.I, rule=activation_rule)

# 6. SOLVE AND OUTPUT
# ===================
import time 
solver = po.SolverFactory('appsi_highs')
try:
    start_time = time.time()
    results = solver.solve(model,tee=True)
    end_time = time.time()
    wall_clock_time = end_time - start_time
    print("\n" + "="*30)
    print(" OPTIMIZATION RESULTS ")
    print("="*30)
    
    total_quality_cost = sum(pyo.value(model.d[i,j]) * pyo.value(model.p[i,j]) for (i,j) in model.ValidPairs)
    active_count = sum(pyo.value(model.y[i]) for i in model.I)
    
    print(f"Total Objective Score: {pyo.value(model.Obj):.4f}")
    print(f"Total Semantic Distance: {total_quality_cost:.4f}")
    print(f"Total Employees Used: {int(active_count)}")
    print("-" * 30)
    print("PERFORMANCE METRICS:")
    # Access time from the results object (solver-reported time)
    # if results.solver.wall_time:
    #     print(f"Solver Reported Time: {results.solver.wall_time:.4f} seconds")
    # Print the manually measured wall-clock time
    print(f"Wall Clock Time:      {wall_clock_time:.4f} seconds")
    print("-" * 30)

    for i in model.I:
        if pyo.value(model.y[i]) > 0.5:
            print(f"\n🟢 Employee {i} (ACTIVE):")
            for j in model.J:
                if (i, j) in model.ValidPairs and pyo.value(model.p[i, j]) > 0.5:
                    dist = distance_map[(i,j)]
                    print(f"   ├── Assigned Task: {j}")
                    print(f"   └── Distance: {dist:.4f}")
                    
except Exception as e:
    print(f"Solver Error: {e}")
    print("Ensure you have GLPK installed (conda install -c conda-forge glpk)")