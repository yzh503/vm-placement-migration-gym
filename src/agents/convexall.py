from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
from src.utils import convert_obs_to_dict
import cvxpy as cvx
from src.vm_gym.envs.env import VmEnv

@dataclass
class ConvexAllConfig: 
    migration_penalty: float = 10000
    pass

class ConvexAllAgent(Base):
    def __init__(self, env: VmEnv, config: ConvexAllConfig):
        super().__init__(type(self).__name__, env, config)
        self.queue = []  # introduce a queue to store operations
        
    def learn(self):
        pass

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        observation = convert_obs_to_dict(self.env.config, observation)
        vm_placement = np.array(observation["vm_placement"])
        vm_cpu = np.array(observation["vm_cpu"])
        vm_memory = np.array(observation["vm_memory"])

        if len(self.queue) > 0:  # check if there's any operation left in the queue
            action = self.queue.pop(0)  # perform the operation and remove it from the queue
            return action + 1
        
        new_placement = self.maximize_nuclear_norm(vm_cpu, vm_memory, vm_placement.copy())

        # figure out VM migrations and add them into queue
        has_migration = False
        migration = new_placement.copy()
        for i in range(len(vm_placement)):
            if vm_placement[i] > -1 and new_placement[i] > -1 and vm_placement[i] != new_placement[i]:
                has_migration = True
                new_placement[i] = -1  # remove the VM from the new placement
        
        if has_migration:
            self.queue.append(migration) 

        return new_placement + 1

    # V is the number of VMs
    # P is the number of PMs
    # A is the 1xV vector of VMs' CPU utilization
    # B is the 1xV vector of VMs' memory utilization
    # X_values is the VxP matrix of VMs' placement
    def maximize_nuclear_norm(self, A, B, vm_placement):

        if np.count_nonzero(vm_placement > -2) == 0:
            return vm_placement
        
        P = self.env.config.pms
        V = self.env.config.vms
        A = A[vm_placement > -2].reshape(1, -1)
        B = B[vm_placement > -2].reshape(1, -1)

        M = np.zeros(shape=(self.env.config.vms, self.env.config.pms))
        for i, pm in enumerate(vm_placement):
            if pm > -1:
                M[i, pm] = 1 
        
        cols_to_optimize = np.ones(P, dtype=bool)
        rows_to_optimize = vm_placement > -2 # variable rows
        rows_optimized = []

        i = 0
        while rows_to_optimize.any() and cols_to_optimize.any():
            cols = np.count_nonzero(cols_to_optimize)
            rows_formatted = []
            # find the first -1 
            variable_rows = [] # only has 1 variable row, becuase multiple variable rows may get stuck in non-existence of solution
            for i, row in enumerate(M[vm_placement > -2]): 
                if rows_to_optimize[i]:
                    Z = cvx.Variable((1, cols))
                    Z.value = row[cols_to_optimize].reshape(1, -1)
                    rows_formatted.append(Z)
                    variable_rows.append(i)
                else:
                    rows_formatted.append(row[cols_to_optimize])

            if len(variable_rows) == 0:
                break
            
            # X is a binary matrix of shape V * P, where V is the number of VMs and P is the number of servers
            # R is resource matrix of shape 2 * P, where P is the number of servers
            # cvx.multiply(M[vm_placement > -2], 1 - X) @ onesn represents if corresponding VM was initially placed on one server and finally re-placed on another server. 
            # onesm @ cvx.multiply(M[vm_placement > -2], 1 - X) @ onesn is the total number of VM migration requests.
            R = np.concatenate((A, B), axis=1)
            X = cvx.bmat(rows_formatted)
            onesm = np.ones(shape=(1, X.shape[0]))
            onesn = np.ones(shape=(1, X.shape[1]))
            constraints = [0 <= X, X <= 1, cvx.sum(X[variable_rows]) == 1, A @ X <= 1, B @ X <= 1]
            objective = cvx.Minimize(cvx.norm(X, 'nuc') + self.config.migration_penalty * onesm @ (cvx.multiply(M[vm_placement > -2][:, cols_to_optimize], 1 - X)) @ onesn.T)
            # (cvx.multiply(M[vm_placement > -2], 1 - X)) @ onesn has shape 1,P

            prob = cvx.Problem(objective, constraints)
            try: 
                prob.solve(solver=cvx.SCS)
            except cvx.SolverError:
                print("SolverError")
                break

            # if prob.status != cvx.OPTIMAL:
            #     #print("No solution on VM", variable_row)
            #     rows_to_optimize[variable_row] = False
            #     continue
            if prob.status != cvx.OPTIMAL:
                #print("No solution on VM ", variable_rows)
                # Reduce the rows to optimise. Reduce the placed VMs first.
                if len(variable_rows) == 1: 
                    rows_to_optimize[variable_rows[0]] = False
                    continue

                migratable_reduced = np.logical_and(rows_to_optimize == True, vm_placement > -1)
                placable_reduced = np.logical_and(rows_to_optimize == True, vm_placement == -1)
                if migratable_reduced.any():
                    reduced = np.where(migratable_reduced)[0][0]
                    rows_to_optimize[reduced] = False
                    #print("Migratable reduced:", reduced)
                elif placable_reduced.any():
                    reduced = np.where(placable_reduced)[0][0]
                    rows_to_optimize[reduced] = False
                    #print("Placable reduced:", reduced)

                continue
            # In some cases there is no solution. For example, if there are too many variable rows and not enough PMs
            """
            X = | 0. 1. |
                | 0. 0. |
                | 0. 0. |
                | Z1    |
                | Z2    |
                | Z3    |
                | Z4    |
                | Z5    |
            """

            X_full = M[vm_placement > -2]
            X_opt = np.array(X.value)

            # Algorithm 2: VM Deployement 
            for v, row in enumerate(X_opt): 
                if rows_to_optimize[v]:
                    p = np.argmax(row).flatten()[0]
                    X_full[v, :] = 0
                    available_pms = np.argwhere(cols_to_optimize == True).flatten()
                    if available_pms.size <= p:
                        continue
                    p_full = available_pms[p]
                    X_full[v, p_full] = 1

                    overloaded = np.logical_or(A @ X_full > 1, B @ X_full > 1)
                    if overloaded.any(): 
                        #print("Overloaded: ", v, p_full, X_full)
                        cols_to_optimize[p_full] = False
                        X_opt = np.delete(X_opt, p, axis=1)
                        X_full[v, :] = M[vm_placement > -2][v, p_full]
                    else: 
                        #print("Underloaded: ", v, p_full, X_full)
                        rows_optimized.append((v, X_full[v]))
                        rows_to_optimize[v] = False

            M[vm_placement > -2] = X_full[:]
        
        for v, row in rows_optimized: 
            pm = np.argwhere(row == 1).flatten() # pm is the index of available PMs
            if pm.size == 1:
                vm_placement[v] = pm[0]
            elif pm.size == 0:
                pass
            else:
                raise Exception("VM is assigned to multiple PMs: ", pm)
            
        return vm_placement
