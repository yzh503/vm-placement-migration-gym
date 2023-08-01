from dataclasses import dataclass, asdict
import numpy as np
from src.agents.base import Base, Config
from src.utils import convert_obs_to_dict
import cvxpy as cvx
from src.vm_gym.envs.env2d import VmEnv

@dataclass
class ConvexConfig(Config): 
    migration_penalty: float = 30
    W: int = 30 
    hard_solution: bool = False

class ConvexAgent(Base):
    def __init__(self, env: VmEnv, config: ConvexConfig):
        super().__init__(type(self).__name__, env, config)
        self.queue = []  # introduce a queue to store operations
        self.no_solution_iter = 0
        
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
        P = self.env.config.pms
        V = self.env.config.vms

        has_migration = False
        while(len(self.queue) > 0):
            has_migration = True
            i, placement = self.queue.pop(0)
            vm_placement[i] = placement
        
        if has_migration:
            return vm_placement

        new_placement = self.maximize_nuclear_norm(P, V, vm_cpu, vm_memory, vm_placement.copy())

        # ExtractVM migrations and add them into the queue
        vm_was_placed = vm_placement < self.env.config.pms
        vm_is_placed = new_placement < self.env.config.pms 
        vm_is_changed = vm_placement != new_placement

        for i in range(len(vm_placement)):
            if vm_was_placed[i] and vm_is_placed[i] and vm_is_changed[i]:
                self.queue.append((i, new_placement[i]))
                new_placement[i] = self.env.config.pms

        return new_placement

    # A is the 1xV vector of VMs' CPU utilization
    # B is the 1xV vector of VMs' memory utilization
    # X_values is the VxP matrix of VMs' placement
    def maximize_nuclear_norm(self, P, V, A, B, vm_placement):

        if (vm_placement > P).all(): # No VM is arrived yet
            return vm_placement

        A = A[vm_placement <= P].reshape(1, -1)
        B = B[vm_placement <= P].reshape(1, -1)

        M = np.zeros(shape=(V, P))
        for i, pm in enumerate(vm_placement):
            if pm < P:
                M[i, pm] = 1 

        cols_to_optimize = np.ones(P, dtype=bool)
        rows_to_optimize = vm_placement <= P # variable rows
        rows_optimized = []
        while rows_to_optimize.any() and cols_to_optimize.any():
            cols = np.count_nonzero(cols_to_optimize)
            rows_formatted = []
            variable_rows = [] # only has 1 variable row, becuase multiple variable rows may get stuck in non-existence of solution
            for i, row in enumerate(M[vm_placement <= P]): 
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
            # cvx.multiply(M[vm_placement <= P], 1 - X) @ onesn represents if corresponding VM was initially placed on one server and finally re-placed on another server. 
            # onesm @ cvx.multiply(M[vm_placement <= P], 1 - X) @ onesn is the total number of VM migration requests.
            X = cvx.bmat(rows_formatted)
            onesm = np.ones(shape=(1, X.shape[0]))
            onesn = np.ones(shape=(1, X.shape[1]))
            constraints = [0 <= X, X <= 1, cvx.sum(X[variable_rows]) == 1, A @ X <= 1, B @ X <= 1]
            objective = cvx.Minimize(cvx.norm(X, 'nuc') + self.config.migration_penalty * onesm @ (cvx.multiply(M[vm_placement <= P][:, cols_to_optimize], 1 - X)) @ onesn.T)

            prob = cvx.Problem(objective, constraints)
            try: 
                prob.solve(solver=cvx.SCS)
            except cvx.SolverError:
                print("SolverError")
                break
            
            # No solution found due to high number of VMs. 
            if prob.status != cvx.OPTIMAL:
                self.no_solution_iter += 1
                if self.config.hard_solution:
                    # Reduce the rows to optimise. Reduce the placed VMs first.
                    if len(variable_rows) == 1: 
                        rows_to_optimize[variable_rows[0]] = False
                        continue

                    migratable_reduced = np.logical_and(rows_to_optimize == True, vm_placement < P)
                    placable_reduced = np.logical_and(rows_to_optimize == True, vm_placement == P)
                    if migratable_reduced.any():
                        reduced = np.where(migratable_reduced)[0][0]
                        rows_to_optimize[reduced] = False
                    elif placable_reduced.any():
                        reduced = np.where(placable_reduced)[0][0]
                        rows_to_optimize[reduced] = False
                    else: 
                        break
                    continue # try again with reduced rows
                else:
                    break

            X_full = M[vm_placement <= P]
            X_opt = np.array(X.value)

            # Algorithm 2: VM Deployement 
            # If the VM placement exceeds the physical limit, remove the PM from the optimisation list.
            sorted_indices = np.argmax(X_opt, axis=1)
            for v, p in enumerate(sorted_indices): 
                if not rows_to_optimize[v]:
                    continue
                X_full[v, :] = 0
                available_pms = np.argwhere(cols_to_optimize == True).flatten()
                if available_pms.size <= p:
                    continue
                p_full = available_pms[p]
                X_full[v, p_full] = 1

                overloaded = np.logical_or(A @ X_full > 1, B @ X_full > 1)
                if overloaded.any(): 
                    cols_to_optimize[p_full] = False
                    X_opt = np.delete(X_opt, p, axis=1)
                    X_full[v, :] = M[vm_placement <= P][v, p_full]
                else: 
                    rows_optimized.append((v, X_full[v]))
                    rows_to_optimize[v] = False

                    # Decision window (maximum number of VMs to be placed)
                    if len(rows_optimized) >= self.config.W:
                        break

            M[vm_placement <= P] = X_full[:]
        
        for v, row in rows_optimized: 
            pm = np.argwhere(row == 1).flatten() # pm is the index of available PMs
            if pm.size == 1:
                vm_placement[v] = pm[0]
            elif pm.size == 0:
                pass
            else:
                raise Exception("VM is assigned to multiple PMs: ", pm)
            
        return vm_placement
