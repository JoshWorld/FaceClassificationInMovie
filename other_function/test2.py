import numpy as np

v = np.array([[0.3, 0.2], [0.8, 0.9], [1.1, 0.9], [1.3, 0.9],
             [0.5, 0.9], [0.6, 0.3], [1.4, 0.7], [1.1, 0.8]])


l = np.array([{'face_idx': 0, 'group_idx': 0, 'e_distance': 0.0016790756955742836, 'p_distance': 0.0, 'sum': 0.0016790756955742836, 'center': np.array([1528,  348]), 'min': np.array([1454,  241]), 'max': np.array([1602,  456])},
     {'face_idx': 0, 'group_idx': 1, 'e_distance': 0.6243399381637573, 'p_distance': 472.9522597472181, 'sum': 473.57659968538184, 'center': np.array([1528,  348]), 'min': np.array([1454,  241]), 'max': np.array([1602,  456])},
     {'face_idx': 0, 'group_idx': 2, 'e_distance': 0.7100513219833374, 'p_distance': 320.32383614086547, 'sum': 321.0338874628488, 'center': np.array([1528,  348]), 'min': np.array([1454,  241]), 'max': np.array([1602,  456])}, {'face_idx': 0, 'group_idx': 3, 'e_distance': 0.34679685831069945, 'p_distance': 178.4, 'sum': 178.74679685831072, 'center': np.array([1528,  348]), 'min': np.array([1454,  241]), 'max': np.array([1602,  456])}, {'face_idx': 1, 'group_idx': 0, 'e_distance': 0.625853419303894, 'p_distance': 472.9522597472181, 'sum': 473.578113166522, 'center': np.array([346, 378]), 'min': np.array([268, 268]), 'max': np.array([425, 488])},
     {'face_idx': 1, 'group_idx': 1, 'e_distance': 0.00289097148925066, 'p_distance': 0.0, 'sum': 0.00289097148925066, 'center': np.array([346, 378]), 'min': np.array([268, 268]), 'max': np.array([425, 488])}, {'face_idx': 1, 'group_idx': 2, 'e_distance': 0.3176024079322815, 'p_distance': 152.8188470052042, 'sum': 153.13644941313646, 'center': np.array([346, 378]), 'min': np.array([268, 268]), 'max': np.array([425, 488])},
     {'face_idx': 1, 'group_idx': 3, 'e_distance': 0.5286439776420593, 'p_distance': 294.6444637185637, 'sum': 295.17310769620576, 'center': np.array([346, 378]), 'min': np.array([268, 268]), 'max': np.array([425, 488])},
     {'face_idx': 2, 'group_idx': 0, 'e_distance': 0.7101266384124756, 'p_distance': 320.32383614086547, 'sum': 321.03396277927794, 'center': np.array([728, 384]), 'min': np.array([643, 269]), 'max': np.array([813, 499])},
     {'face_idx': 2, 'group_idx': 1, 'e_distance': 0.31892781257629393, 'p_distance': 152.8188470052042, 'sum': 153.1377748177805, 'center': np.array([728, 384]), 'min': np.array([643, 269]), 'max': np.array([813, 499])},
     {'face_idx': 2, 'group_idx': 2, 'e_distance': 0.00018885688623413444, 'p_distance': 0.0, 'sum': 0.00018885688623413444, 'center': np.array([728, 384]), 'min': np.array([643, 269]), 'max': np.array([813, 499])},
     {'face_idx': 2, 'group_idx': 3, 'e_distance': 0.5427912354469299, 'p_distance': 142.33032003055428, 'sum': 142.87311126600122, 'center': np.array([728, 384]), 'min': np.array([643, 269]), 'max': np.array([813, 499])},
     {'face_idx': 3, 'group_idx': 0, 'e_distance': 0.3513102293014526, 'p_distance': 178.4, 'sum': 178.75131022930145, 'center': np.array([1082,  348]), 'min': np.array([1008,  239]), 'max': np.array([1157,  458])},
     {'face_idx': 3, 'group_idx': 1, 'e_distance': 0.5196796774864196, 'p_distance': 294.6444637185637, 'sum': 295.1641433960501, 'center': np.array([1082,  348]), 'min': np.array([1008,  239]), 'max': np.array([1157,  458])},
     {'face_idx': 3, 'group_idx': 2, 'e_distance': 0.5345129370689392, 'p_distance': 142.33032003055428, 'sum': 142.86483296762322, 'center': np.array([1082,  348]), 'min': np.array([1008,  239]), 'max': np.array([1157,  458])},
     {'face_idx': 3, 'group_idx': 3, 'e_distance': 0.032808443158864976, 'p_distance': 0.0, 'sum': 0.032808443158864976, 'center': np.array([1082,  348]), 'min': np.array([1008,  239]), 'max': np.array([1157,  458])}])

print(l)
m = np.array([ item['sum'] for item in l])
m = (m - np.mean(m))/np.std(m)
print(m)
