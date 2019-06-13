import scipy.io as sio

mat = sio.loadmat('cars_annos.mat')
for class_names in mat['class_names']:
    class_name_index = []
    for x in class_names:
        class_name_index.append(x[0])
