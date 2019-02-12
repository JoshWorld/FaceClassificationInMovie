import numpy as np


def calc_vector_distance(v1, v2):
    dist = np.linalg.norm(v1 - v2)
    return dist


A = np.random.random((10, 2))
B = np.random.random((5, 2))

print(A)
print('--')
print(B)


def calc_min_distance(v1_list, v2_list):
    match_list = []
    for i in range(1, len(v1_list)):
        min_dist = calc_vector_distance(A[0], B[0])
        min_idx = 0
        for j in range(1, len(v2_list)):
            dist = calc_vector_distance(A[i], B[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        match_list.append({'match_index':(i, min_idx),'min_dist':min_dist})

    # Normalization
    x = np.array([item['min_dist'] for item in match_list])
    y = x / sum(x)

    for item, norm_dist in zip(match_list, y):
        item['norm_min_dist'] = norm_dist
    return match_list


print(calc_min_distance(A, B))

