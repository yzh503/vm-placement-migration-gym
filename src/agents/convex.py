from dataclasses import dataclass, asdict
import numpy as np
from src.agents.base import Base, Config
from src.utils import convert_obs_to_dict
import cvxpy as cvx
from vmenv.envs.env import VmEnv
import threading

@dataclass
class ConvexConfig(Config): 
    W: int = 30 
    frequency: int = 3
    timeout: int = 3

class ConvexAgent(Base):
    def __init__(self, env: VmEnv, config: ConvexConfig):
        super().__init__(type(self).__name__, env, config)
        self.queue = []  # introduce a queue to store operations
        self.failures = 0
    
    def eval(self):
        pass

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
        
        to_skip = self.env.timestep % self.config.frequency > 0 and self.env.config.eval_steps != self.env.timestep
        if has_migration or to_skip:
            return vm_placement
        
        result_container = {"result": None}

        result_container["result"] = self.maximize_nuclear_norm(P, V, vm_cpu, vm_memory, vm_placement.copy())

        def threaded_function():
            result_container["result"] = self.maximize_nuclear_norm(P, V, vm_cpu, vm_memory, vm_placement.copy())

        thread = threading.Thread(target=threaded_function)
        thread.start()
        thread.join(timeout=60)

        if thread.is_alive():
            raise Exception(f"Optimisation timeout: {self.env.config.seed} at {self.env.timestep}")
   
        new_placement = result_container["result"]

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

        M = np.zeros(shape=(V, P))
        for i, pm in enumerate(vm_placement):
            if pm < P:
                M[i, pm] = 1 

        cols_to_optimize = np.ones(P, dtype=bool) 
        rows_to_optimize = vm_placement <= P 

        rows_optimized = []
        while rows_to_optimize.any() and cols_to_optimize.any():
            cols = np.count_nonzero(cols_to_optimize)
            rows_formatted = []
            S = np.zeros_like(M[:, cols_to_optimize])

            optimising = []
            for i, row in enumerate(M[:, cols_to_optimize]): 
                if rows_to_optimize[i]:
                    Z = cvx.Variable((1, cols), integer=True)
                    Z.value = row.reshape(1, -1)
                    rows_formatted.append(Z)
                    S[i, :] = 1
                    optimising.append(i)
                elif vm_placement[i] <= P: # ignore unarrived VMs
                    rows_formatted.append(row.reshape(1, -1))

            if len(optimising) <= 1:
                return vm_placement

            # X is a binary matrix of shape V * P, where V is the number of VMs and P is the number of servers
            # R is resource matrix of shape 2 * P, where P is the number of servers
            # cvx.multiply(M[vm_placement <= P], 1 - X) @ onesn represents if corresponding VM was initially placed on one server and finally re-placed on another server. 
            # onesm @ cvx.multiply(M[vm_placement <= P], 1 - X) @ onesn is the total number of VM migration requests.
            X = cvx.bmat(rows_formatted)

            Am = A[vm_placement <= P].reshape(1, -1)
            Bm = B[vm_placement <= P].reshape(1, -1)
            ones = np.ones((cols, 1))
            constraints = [
                X >= 0,
                X <= 1,
                X @ ones <= 1,
                Am @ X <= 1,
                Bm @ X <= 1
            ]
            objective = cvx.Minimize(cvx.sum(-X)) # Rank minimization is unsolvable if PMs are insufficient
            problem = cvx.Problem(objective, constraints)

            try: 
                problem.solve(solver=cvx.SCIP, scip_params={"limits/time": self.config.timeout})
            except cvx.SolverError as e:
                print(e)    
                break
 
            # No solution found due to high number of VMs. 
            if problem.status != cvx.OPTIMAL:
                break

            X_full = M[vm_placement <= P].copy() # use vm_placement <= P here because rows_to_optimize is reducing
            X_opt = np.array(X.value)

            # Algorithm 2: VM Deployement 
            # For each placement, if the VM placement exceeds the physical limit, remove the PM from the optimisation list.
            sorted_indices = np.argmax(X_opt, axis=1)
            for v, p in enumerate(sorted_indices): 
                if not rows_to_optimize[v]:
                    continue
                
                # each placement is done for a VM, clear the vm placement. 
                X_full[v, :] = 0
                available_pms = np.argwhere(cols_to_optimize == True).flatten()
                if available_pms.size <= p:
                    continue
                p_full = available_pms[p]
                X_full[v, p_full] = 1

                # Check if the PM is overloaded
                overloaded = np.logical_or(Am @ X_full > 1, Bm @ X_full > 1)
                if overloaded.any(): 
                    cols_to_optimize[p_full] = False
                    X_opt = np.delete(X_opt, p, axis=1)
                    X_full[v, :] = M[vm_placement <= P][v, p_full]
                else: 
                    rows_optimized.append((v, X_full[v]))
                    rows_to_optimize[v] = False

                    # Decision window (maximum number of VMs to be placed)
                    if len(rows_optimized) >= self.config.W:
                        rows_to_optimize[:] = False
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
