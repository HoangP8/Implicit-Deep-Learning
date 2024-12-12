

def solve(X, U, Z, Y, config):
    if config.solver=="mosek":
        from sim.solvers.mosek import parallel_solve
    elif config.solver == "admm_constraint_consensus":
        from sim.solvers.admm_constraint_consensus import parallel_solve
    elif config.solver == "state_matching":
        from sim.solvers.state_matching import parallel_solve
    elif config.solver == "lowrank":
        from sim.solvers.lowrank import parallel_solve
    else:
        raise NotImplementedError()
    
    return parallel_solve(X, U, Z, Y, config)