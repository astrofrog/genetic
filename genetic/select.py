import random


def select(chi2, n, k_frac, p):
    """
    Tournament selection routine
    Good parameters for getting ~10% are k_frac=0.2 and p=0.9
    """

    k = int(len(chi2) * k_frac)

    assert k > 0, "k_frac is too small"

    model_id = [i for i in range(len(chi2))]

    prob = [p * (1 - p) ** j for j in range(k)]
    norm = sum(prob)
    for i in range(len(prob)):
        prob[i] = prob[i] / norm

    choices = []

    for t in range(n):

        pool_id = random.sample(model_id, k)

        pool_chi = chi2[pool_id]

        aux_list = zip(pool_chi, pool_id)
        aux_list.sort()
        pool_chi, pool_id = map(list, zip(*aux_list))

        xi = random.random()
        for j in range(k):
            if(xi <= sum(prob[0: j + 1])):  # is  + 1 because prob[0: 0] is empty
                choice = pool_id[j]
                choices.append(choice)
                break

    return(choices)
