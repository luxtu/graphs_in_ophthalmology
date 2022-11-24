import numpy as np




def max_assign(idcs, maskList):
    max_val = 0
    max_idx = -1
    for i, mask in enumerate(maskList):
        val = np.sum(mask[idcs[0]: idcs[1], idcs[2]: idcs[3], idcs[4]:idcs[5]])
        if val > max_val:
            max_val = val 
            max_idx = i 
        elif val <0:
            print(val)

    return max_idx





def mix_assign(idcs, maskList):
    rel_values = []
    for i, mask in enumerate(maskList):
        val = np.sum(mask[idcs[0]: idcs[1], idcs[2]: idcs[3], idcs[4]:idcs[5]])
        if val > 0:
            rel_values.append(i)

    max_idx = sum(rel_values)
    if len(rel_values)>1:
        max_idx += len(rel_values)-1


    return max_idx