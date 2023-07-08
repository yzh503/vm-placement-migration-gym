from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
from src.utils import convert_obs_to_dict
import cvxpy as cvx
from src.vm_gym.envs.env import VmEnv

@dataclass
class ConvexConfig: 
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
        for i in range(len(vm_placement)):
            if vm_placement[i] > -1 and new_placement[i] > -1 and vm_placement[i] != new_placement[i]:
                self.queue.append(new_placement.copy()) 
                new_placement[i] = -1  # remove the VM from the new placement

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
            for i, row in enumerate(M[vm_placement > -2]): 
                if rows_to_optimize[i]:
                    Z = cvx.Variable((1, cols))
                    Z.value = row[cols_to_optimize].reshape(1, -1)
                    rows_formatted.append(Z)
                else:
                    rows_formatted.append(row[cols_to_optimize])
            X = cvx.bmat(rows_formatted)
            print(X.value)
            ones = np.ones(cols).reshape(1, -1)
            constraints = [0 <= X, X <= 1, ones @ X.T == 1, A @ X <= 1, B @ X <= 1]
            objective = cvx.Minimize(cvx.norm(X, 'nuc'))

            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert prob.status == cvx.OPTIMAL, prob.status

            X_full = M[vm_placement > -2]
            X_opt = np.array(X.value)

            # Algorithm 2: VM Deployement 
            for v, row in enumerate(X_opt): 
                if rows_to_optimize[v]:
                    p = np.argmax(row).flatten()[0]
                    X_full[v, :] = 0
                    available_pms = np.argwhere(cols_to_optimize == True).flatten()
                    if available_pms.size == 0:
                        break
                    p_full = available_pms[p]
                    X_full[v, p_full] = 1
                    overloaded = np.logical_or(A @ X_full > 1, B @ X_full > 1)
                    if np.count_nonzero(overloaded) > 0: 
                        cols_to_optimize[p_full] = False
                        X_opt = np.delete(X_opt, p, axis=1)
                    else: 
                        rows_optimized.append((v, X_full[v]))
                        rows_to_optimize[v] = False
                    X_opt[v, :] = 0

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
