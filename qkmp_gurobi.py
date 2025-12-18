import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
#parameter declarations
n=6 #item
m=2 #knapsack
print("\weight")
w=[]
for i in range(n):
    w.append(random.randint(10, 50))
print(w)
#profit for n items
print("\nprofit per item:")
p = []
for i in range(n):
    p.append(random.randint(10, 50)*10)
print(p)
# capacity of m knapsacks
c = [100, 50]
# profit for every pair of items
print("\nprofit per pair of items:")
q = []
for i in range(n):
    qq = []
    for j in range(n):
        qq.append(0) # qq.append(random.randint(10, 50)*10)
    q.append(qq)
for i in range(n-1):
    # q[i][i] = random.randint(10, 50)*10
    for j in range(i+1, n):
        q[i][j] = random.randint(10, 50) * 10
        q[j][i] = q[i][j]
try:
  model = gp.Model(name='qkmp')
  #decision variables
  x=np.empty(n, dtype=object) 
  for i in range(n):
    x[i]=np.empty(m,dtype=object)
    for j in range(m):
        x[i][j]=model.addVar(vtype=GRB.BINARY,name='x'+str(i+1)+str(j+1))
  #constraints
  #capacity
  for j in range(m):
    model.addConstr(sum(w[i]*x[i][j] for i in range(n))<=c[j])
  
  #one item one knapsack
  for i in range(n):
    model.addConstr(sum(x[i][j] for j in range(m))<=1)
  
  #objective function
  model.setObjective(sum(sum(p[i]*x[i][j] for i in range(n)) + 
                    sum((sum(q[i][k]*x[i][j]*x[k][j] for k in range(i+1,n))) for i in range(n-1)) for j in range(m)),GRB.MAXIMIZE)
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
  