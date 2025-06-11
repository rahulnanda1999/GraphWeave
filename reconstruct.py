import cvxpy as cp
import numpy as np
import scipy
from pandas import Series, DataFrame
import pandas as pd

def get_LHS_RHS(V_t, V_tplus1, alpha, degrees):
  deg_prime = alpha + (1-alpha)*degrees
  deg_prime_sqrt = np.sqrt(deg_prime)
  LHS = (1-alpha) * V_t * (1/deg_prime_sqrt)[None,:]
  RHS = V_tplus1 * deg_prime_sqrt[None,:] - alpha * V_t * (1/deg_prime_sqrt)[None,:]
  return LHS, RHS

def reconstruct_G_using_D(V_t, V_tplus1, alpha, degrees, indices=None, max_time=120, MIPGap=0.05):
  n = V_t.shape[1]
  A = cp.Variable((n,n), symmetric=True)

  constraints = [A >= 0]
  constraints += [A <= 1]  

  if indices is not None:
    constraints += [A[indices] == 0]

  constraints += [cp.sum(A[i])==degrees[i] for i in range(n)]  
  constraints += [cp.diag(A)==0]

  LHS, RHS = get_LHS_RHS(V_t, V_tplus1, alpha, degrees)

  obj = cp.Minimize(cp.sum(cp.abs(LHS @ A - RHS)))

  prob = cp.Problem(obj, constraints)
#  prob.solve()
#  prob.solve(solver=cp.SCIPY, scipy_options={'method':'highs-ipm'})
#  prob.solve(solver=cp.MOSEK, accept_unknown=True)
  prob.solve(solver=cp.GUROBI, TimeLimit=max_time, MIPGap=MIPGap)

  print(f'status={prob.status}, objective={prob.value:2.2f}, solve time={prob.solver_stats.solve_time:1.1f}s')
  return A.value

def reconstruct_G_using_D_integer(V_t, V_tplus1, alpha, degrees, indices=None, max_time=120, MIPGap=0.1):
  n = V_t.shape[1]
  A = cp.Variable((n,n), boolean=True)

  constraints = []
  constraints += [A == A.T]

  if indices is not None:
    constraints += [A[indices] == 0]

  constraints += [cp.sum(A[i])==degrees[i] for i in range(n)]
  constraints += [cp.diag(A)==0]

  LHS, RHS = get_LHS_RHS(V_t, V_tplus1, alpha, degrees)

  obj = cp.Minimize(cp.sum(cp.abs(LHS @ A - RHS)))

  prob = cp.Problem(obj, constraints)
#  prob.solve()
#  prob.solve(solver=cp.SCIPY, scipy_options={'method':'highs-ipm'})
#  prob.solve(solver=cp.MOSEK, verbose=verbose, accept_unknown=True, mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME':max_time})
  prob.solve(solver=cp.GUROBI, TimeLimit=max_time, MIPGap=MIPGap)

  print(f'status={prob.status}, objective={prob.value:2.2f}, solve time={prob.solver_stats.solve_time:1.1f}s')
  return A.value

def pick_threshold5(A_pred, degrees):
  f = lambda x: np.mean(np.abs((A_pred>x[0]+x[1]*np.log(degrees)[:,None]).sum(axis=1) / degrees - 1))
  a_list = (0,1)
  b_list = (0,1/np.log(max(degrees)))
  x = scipy.optimize.brute(f, (a_list, b_list), Ns=100)
  A_pred2 = (A_pred>x[0]+x[1]*np.log(degrees)[:,None]).astype(int)
  return A_pred2


