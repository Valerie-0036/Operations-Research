import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random 
#parameter declarations
n=4 #city
c=np.array([[0,10,12,20],[10,0,17,15],[12,17,0,11],[20,15,11,0]]) #cost
try:
  model = gp.Model(name='TSP')
  #decision variables
  x=np.empty((n,n), dtype=object) 
  u=np.empty(n, dtype=object) 
  for i in range(n):
    u[i]=model.addVar(vtype=GRB.INTEGER,name='u' + str(i))
    for j in range(n):
      x[i,j]=model.addVar(vtype=GRB.BINARY, name='x'+str(i+1)+str(j+1))
  #constraints
  #each city mus be visited
  for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n) if i!=j)==1)
  for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n) if j!=i)==1)
  # a single tour
  for i in range(1,n):
    for j in range(1,n):
      if(i!=j):
        model.addConstr(u[i]-u[j]+1<=(n-1)*(1-x[i,j]))

  for i in range(1,n):
    model.addConstr(u[i]>=2)
    model.addConstr(u[i]<=n)
  #objective function
  model.setObjective(sum(sum(c[i,j]*x[i,j] for j in range(n))for i in range(n)),GRB.MINIMIZE)
  model.optimize()
  if model.status ==GRB.OPTIMAL:
    print('Optimal Obj: %g' % model.ObjVal)
    for v in model.getVars(): 
      if(v.X!=0):
        print('%s %g' % (v.VarName, v.X))
  else:
    print('Optimization ended with status %d' %model.status)
    sys.exit(0)
except gp.GurobiError as e:
  print(f"Error code {e.errno}: {e}")
except AttributeError:
  print('Encountered an attribute error')
  