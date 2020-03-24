

def label_filter(tmplt, world, candidates, *, verbose=False, **kwargs):
    candidates[:,:] &= tmplt.labels.reshape(-1,1) == world.labels.reshape(1,-1)
    return tmplt, world, candidates
