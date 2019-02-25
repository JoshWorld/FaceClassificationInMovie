import numpy as np
import pickle
from other_function import get_embedding_vector

f = open('../face_data.save','rb')
a = pickle.load(f)

data = []
Y = []
for group in a:
    for item in group:
        try:
            Y.append(item['group_idx'])
            data.append(item['embedding_vector'])
        except:
            pass


X = np.array(data).squeeze()
y = np.array(Y)

from sklearn.svm import SVC

clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)


v = get_embedding_vector.get_embedding_vector_func('C:\\Users\\ADMIN\\Desktop\\FaceClassificationInMovie\\test_data/image\\test5.JPG')

print(clf.predict([np.array(v).squeeze()]))