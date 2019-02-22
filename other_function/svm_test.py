import numpy as np
import pickle
from other_function import get_embedding_vector

f = open('../face_data.save','rb')
a = pickle.load(f)

print(a)

data = []
Y = []
for group in a:
    for item in group:

        try:
            Y.append(item['group_idx'])
            data.append(item['embedding_vector'])
        except:
            pass

print(len(data), len(Y))
X = np.array(data).squeeze()
y = np.array(Y)

print(y)

from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf.fit(X, y)

SVC(C=1.0, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=128, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

v = get_embedding_vector.get_embedding_vector_func('C:\\Users\\ADMIN\\Desktop\\FaceClassificationInMovie\\test_data/image\\test5.JPG')
print(v)
print(clf.predict([np.array(v).squeeze()]))