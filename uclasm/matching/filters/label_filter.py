def label_filter(exact_smp, *, verbose=False, **kwargs):
    tmplt = exact_smp.tmplt
    world = exact_smp.world
    if hasattr(tmplt, 'labels') and hasattr(world, 'labels'):
        candidates = exact_smp.candidates()
        candidates[:,:] &= tmplt.labels.reshape(-1,1) == world.labels.reshape(1,-1)
