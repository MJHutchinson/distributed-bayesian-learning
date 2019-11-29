import numpy as np

def make_split_indicies(
    num_points,
    shard,
    num_shards,
    seed,
    num_validation,
    num_test
):
    np.random.seed(seed)

    indicies = np.random.permutation(num_points)

    num_train = num_points - num_validation - num_test

    if shard == 'validation':
        return indicies[num_train : num_train + num_validation]
    elif shard == 'test':
        return indicies[num_train + num_validation :]
    elif shard < num_shards:
        points_per_shard = int(np.floor(num_train / num_shards))
        return indicies[shard * points_per_shard : (shard + 1 ) * points_per_shard]
    else:
        raise ValueError('Shard should be less than num shards, validation or test')