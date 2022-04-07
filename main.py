import numpy as np
from svm2 import SVM
from load_data_2 import Load_Data_2
from load_data import LOAD_DATA
from matplotlib import pyplot as plt

if __name__ == '__main__':
    my_load_data = Load_Data_2()
    X, XL, T, TL = my_load_data.load_Iris_2()
    XL[XL == 1] = 1
    XL[XL == 2] = -1
    XL[XL == 0] = -1
    TL[TL == 1] = 1
    TL[TL == 2] = -1
    TL[TL == 0] = -1
    my_svm = SVM(X, XL, T, TL, 0.001)
    p = 3
    # print(my_svm.test_model(0))
    my_svm.train_svm(p)
    my_svm.process_of_test_data(p)
    print(my_svm.get_result(p))
    print(my_svm.test_model(p))
    print(TL)
    # print(w, b)
    # x = np.arange(0, 10, 0.1)
    # y = - b / w[1] - (w[0] / w[1]) * x
    # plt.plot(x, y)
    # plt.scatter(X[:, 0:1], X[:, 1:], c=XL)
    # plt.show()
