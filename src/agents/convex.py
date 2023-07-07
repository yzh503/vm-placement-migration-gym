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
            return action

        new_placement = self.maximize_nuclear_norm(vm_cpu, vm_memory, vm_placement.copy())

        # figure out VM migrations and add them into queue
        for i in range(len(vm_placement)):
            if vm_placement[i] > -1 and new_placement[i] > -1 and vm_placement[i] != new_placement[i]:
                self.queue.append((i, -1))  # remove operation
                self.queue.append((i, new_placement[i]))  # place operation

        if len(self.queue) > 0:
            action = self.queue.pop(0)  # if we have added some operations into the queue, perform the first one
        else:
            action = new_placement + 1  # if no operations were added, just apply the new placement

        return action

    # V is the number of VMs
    # P is the number of PMs
    # A is the 1xV vector of VMs' CPU utilization
    # B is the 1xV vector of VMs' memory utilization
    # X_values is the VxP matrix of VMs' placement
    def maximize_nuclear_norm(self, A, B, vm_placement):

        V = self.env.config.vms
        P = self.env.config.pms

        M = np.zeros(shape=(self.env.config.vms, self.env.config.pms))
        for i, pm in enumerate(vm_placement):
            if pm > -1:
                M[i, pm] = 1 
        
        cols_to_optimize = np.ones(P, dtype=bool)
        rows_to_optimize = np.argwhere(vm_placement > -2).flatten().tolist()
        rows_optimized = []
        while len(rows_to_optimize) > 0:
            i = 0
            while i < len(rows_to_optimize):
                v = rows_to_optimize[i]
                var_columns = M[v, cols_to_optimize]
                X = cvx.Variable((1, var_columns.size))
                X.value = var_columns.reshape(1, -1)

                if v == 0:
                    rows = [X] + [M[1:, cols_to_optimize]]
                elif v == V - 1:
                    rows = [M[:v, cols_to_optimize]] + [X]
                else: 
                    rows = [M[:v, cols_to_optimize]] + [X] + [M[v+1:, cols_to_optimize]]

                rows_formatted = []
                for r in rows:
                    if isinstance(r, cvx.Variable):
                        rows_formatted.append(r)  # cvxpy Variables can be added as is
                    else:
                        rows_formatted.extend(r.tolist())  # Numpy arrays need to be turned into lists

                X = cvx.bmat(rows_formatted)
                constraints = [0 <= X, X <= 1, X.T @ np.ones(V) == 1, A @ X <= 1, B @ X <= 1]
                objective = cvx.Minimize(cvx.norm(X, 'nuc'))

                prob = cvx.Problem(objective, constraints)
                prob.solve(solver=cvx.CVXOPT)

                # For each row of X (VM), change the highest value to 1 and other values to 0
                N = np.array(X.value)

                max_index = np.argmax(N[v])
                N[v, :] = 0
                N[v, max_index] = 1

                N_full = np.copy(M)  # Start from a copy of M
                N_full[v, cols_to_optimize] = N[v, cols_to_optimize] 

                # Exclude PMs that are or will be overloaded
                underloaded = np.logical_and(A @ N_full <= 1, B @ N_full <= 1)
                if np.count_nonzero(underloaded) == len(underloaded):
                    rows_optimized.append(N_full[v])
                    del rows_to_optimize[i]
                else: 
                    i += 1

                cols_to_optimize = np.logical_and(underloaded, cols_to_optimize)

                # Update placement
                M = np.copy(N_full)
        
        for row in rows_optimized: 
            pm = np.argwhere(row == 1).flatten() # pm is the index of available PMs
            if pm.size == 1:
                vm_placement[v] = pm[0]
            elif pm.size == 0:
                pass
            else:
                raise Exception("VM is assigned to multiple PMs: ", pm)
        return vm_placement
