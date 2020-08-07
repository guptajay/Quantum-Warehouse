# Baseline Policy
def baseline_policy(obs):
    """
    Selects the outermost shelf available in the warehouse.

    Args:
        obs (dictionary): Contains warehouse state returned by the environment.
    """
    shelves = []
    for i in range(7 * 7):
        shelves.append(obs[i+1]['status'])

    for i in range(7 * 7):
        if(shelves[i] == 0):
            return i+1
