from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
from src.utils import convert_obs_to_dict
import cvxpy as cvx
from src.vm_gym.envs.env import VmEnv

@dataclass
class ConvexConfig: 
    migration_penalty: float = 1
    pass

class ConvexAgent(Base):
    def __init__(self, env: VmEnv, config: ConvexConfig):
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
        while np.count_nonzero(rows_to_optimize) > 0 and np.count_nonzero(cols_to_optimize) > 0:
            cols = np.count_nonzero(cols_to_optimize)
            rows_formatted = []
            # find the first -1 
            variable_row = None # only has 1 variable row, becuase multiple variable rows may get stuck in non-existence of solution
            for i, row in enumerate(M[vm_placement > -2]): 
                if rows_to_optimize[i] and variable_row is None:
                    Z = cvx.Variable((1, cols))
                    Z.value = row[cols_to_optimize].reshape(1, -1)
                    rows_formatted.append(Z)
                    variable_row = i
                else:
                    rows_formatted.append(row[cols_to_optimize])

            if variable_row is None: 
                break
            
            # X is a binary matrix of shape V * P, where V is the number of VMs and P is the number of servers
            # R is resource matrix of shape 2 * P, where P is the number of servers
            # cvx.multiply(M[vm_placement > -2], 1 - X) @ onesn represents if corresponding VM was initially placed on one server and finally re-placed on another server. 
            # onesm @ cvx.multiply(M[vm_placement > -2], 1 - X) @ onesn is the total number of VM migration requests.
            R = np.concatenate((A, B), axis=1)
            X = cvx.bmat(rows_formatted)
            onesm = np.ones(shape=(1, X.shape[0]))
            onesn = np.ones(shape=(1, X.shape[1]))
            constraints = [0 <= X, X <= 1, cvx.sum(X[variable_row, :]) == 1, A @ X <= 1, B @ X <= 1]
            objective = cvx.Minimize(cvx.norm(X, 'nuc') + self.config.migration_penalty * onesm @ (cvx.multiply(M[vm_placement > -2][:, cols_to_optimize], 1 - X)) @ onesn.T)
            # (cvx.multiply(M[vm_placement > -2], 1 - X)) @ onesn has shape 1,P

            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CVXOPT)
            # Timestep: 		47
            # VM request: 		1, dropped: 0
            # VM placement: 		[ 8  1  8  9  6  4  7 -1 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2]
            # VM suspended: 		[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            # CPU (%): 		[ 0 34  0  0 92  0 83 65 81 11] 3.66
            # Memory (%): 		[ 0 96  0  0 48  0 38 84 79 95] 4.4
            # VM CPU (%): 		[67 34 14 11 83 92 65 76  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 4.42
            # VM Memory (%): 		[56 96 23 95 38 48 84 47  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 4.87
            # VM waiting time: 	[2 5 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            # VM planned runtime: 	[ 84 111  85 100 108  98  99  94   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
            # VM remaining runtime: 	[57 92 63 83 92 86 93 94  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
            
            # column 5 8 9 are available PMs, 
            # VM 7 should be placed on PM 5, 
            # However, VM 7 is skipped
            # X[6] = array([1., 0., 0.]), which means VM 6 is placed on PM 5
            # However, 
            if prob.status != cvx.OPTIMAL:
                #print("No solution on VM", variable_row)
                rows_to_optimize[variable_row] = False
                continue
            # if prob.status != cvx.OPTIMAL:
            #     # Reduce the rows to optimise. Reduce the placed VMs first.
            #     to_disable = np.logical_and(rows_to_optimize == True, vm_placement > -1)
            #     last_rows_to_optimize = np.argwhere(to_disable == True).flatten()
            #     if last_rows_to_optimize.size == 0:
            #         print("cannot reduce: ", rows_to_optimize)
            #         break
            #     else: 
            #         last_row_to_optimize = last_rows_to_optimize[-1]
            #         rows_to_optimize[last_row_to_optimize] = False
            #         print("reduced: ", rows_to_optimize)
            #         continue 
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
