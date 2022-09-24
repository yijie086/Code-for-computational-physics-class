import numpy as np
import pickle as pickle
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import time
import warnings

# np.random.seed()  # shuffle random seed generator

# Ising model parameters
L = 40  # linear system size
J = -1.0  # Ising interaction
T = np.linspace(0.25, 4.0, 16)  # set of temperatures
T_c = 2.26  # Onsager critical temperature in the TD limit

# print("data import begin!")
f = open("Ising2DFM_reSample/Ising2DFM_reSample_L40_T%3DAll.pkl", "rb")
data = pickle.load(f)
data = np.unpackbits(data).reshape(-1, 1600)
label = pickle.load(open("Ising2DFM_reSample/Ising2DFM_reSample_L40_T%3DAll_labels.pkl", "rb"))
print("data import end!")

# print("data divide begin!")
# divide data into ordered, critical and disordered
X_ordered = data[:70000, :]
Y_ordered = label[:70000]
# print("data divide ordered end!")

X_critical = data[70000:100000, :]
Y_critical = label[70000:100000]
# print("data divide critical end!")

X_disordered = data[100000:, :]
Y_disordered = label[100000:]
# print("data divide disordered end!")
# print("data divide end!")

del data, label
# print("data temple del!")

train_test_ratio = 0.9
test_train_ratio = 1-train_test_ratio
# print("training and test data ratio set!")
print("training_to_test ratio: %f" % train_test_ratio)

# print("define training and test data sets begin!")
X = np.concatenate((X_ordered, X_disordered))
Y = np.concatenate((Y_ordered, Y_disordered))

# print("pick random data points to create the training and test sets")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_test_ratio, test_size=test_train_ratio)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print("training and test set end!")

# print("plot a few Ising states begin!")
# set colour bar map
# colormap_args = dict(cmap='plasma_r')
# fig = plt.figure()
# ax = plt.axes()

# plt.imshow(X_ordered[20001].reshape(L, L), **colormap_args)
# plt.title('ordered phase')
# plt.tick_params(labelsize=16)
# plt.show()

# plt.imshow(X_critical[10001].reshape(L, L), **colormap_args)
# plt.title('critical phase')
# plt.tick_params(labelsize=16)
# plt.show()

# plt.imshow(X_disordered[50001].reshape(L, L), **colormap_args)
# plt.title('disordered phase')
# plt.tick_params(labelsize=16)
# plt.show()
# print("plot a few Ising states end!")

# print("apply Random Forest begin")

warnings.filterwarnings("ignore")
# Comment to turn on warnings

nest_test = 20
n_depth = 10
n_sample_leaf = 10
# print("tree para set end!")

myRF_clf = RandomForestClassifier(
        n_estimators=nest_test,
        max_depth=n_depth,
        min_samples_split=n_sample_leaf,  # minimum number of sample per leaf
        oob_score=True,
        random_state=0,
        warm_start=True  # this ensures that you add estimators without retraining everything
    )

print('t_num: %i, t_dep: %i, sample/leaf: %i' % (myRF_clf.n_estimators, myRF_clf.max_depth, myRF_clf.min_samples_split))
# print("RFC train begin!")
start_time = time.time()
myRF_clf.fit(X_train, Y_train)
run_time = time.time() - start_time
RFC_train_accuracy = myRF_clf.score(X_train, Y_train)
RFC_OOB_accuracy = myRF_clf.oob_score_
RFC_test_accuracy = myRF_clf.score(X_test, Y_test)
RFC_critical_accuracy = myRF_clf.score(X_critical, Y_critical)
result = (run_time, RFC_train_accuracy, RFC_OOB_accuracy, RFC_test_accuracy, RFC_critical_accuracy)
print('{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}'.format("time/s", "train score", "OOB estimate", "test score", "crit score"))
print('{0:<15.4f}{1:<15.4f}{2:<15.4f}{3:<15.4f}{4:<15.4f}'.format(*result))
print("RFC train end!")
print("%f\t%f\t%f\t%f" % (RFC_train_accuracy, RFC_OOB_accuracy, RFC_test_accuracy, RFC_critical_accuracy))
exit()
