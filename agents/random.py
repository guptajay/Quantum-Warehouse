from random import randint


def random_policy(obs):
    """
    Selects a random shelve in the warehouse.

    Args:
        obs (dictionary): Contains warehouse state returned by the environment.
    """
    action = randint(1, 49)
    # Check if the action will overwrite a package in existing location
    validAction = False
    while not validAction:
        # take a random action again if a package is already there
        if(obs[action]['status']):
            action = randint(1, 49)
        else:
            validAction = True

    return action
