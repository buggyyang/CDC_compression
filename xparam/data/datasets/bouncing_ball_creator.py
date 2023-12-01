"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

from numpy import *
from scipy import *
import pdb
import pickle
import scipy.io
import sys, os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

shape_std = shape


def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()


size_std = size


def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()


det = linalg.det


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x): return sqrt((x ** 2).sum())


def sigmoid(x):        return 1. / (1. + exp(-x))


SIZE = 10


# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None: r = array([1.2] * n)
    if m is None: m = array([1] * n)
    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    v = randn(n, 2)
    v = v / norm(v) * .5
    good_config = False
    while not good_config:
        x = 2 + rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:      good_config = False
                if x[i][z] + r[i] > SIZE:     good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t, i] = x[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:  v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE: v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def matricize(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None: r = array([1.2] * n)

    A = zeros((T, res, res), dtype='float')

    [I, J] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t] += exp(-(((I - X[t, i, 0]) ** 2 + (J - X[t, i, 1]) ** 2) / (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r == None: r = array([1.2] * n)
    x = bounce_n(T, n, r);
    A = matricize(x, res, r)
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r == None: r = array([1.2] * n)
    x = bounce_n(T, n, r, m);
    V = matricize(x, res, r)
    return V.reshape(T, res ** 2)


# make sure you have this folder
# logdir = './sample'


# def show_sample(V):
#     T = len(V)
#     res = int(sqrt(shape(V)[1]))
#     for t in range(T):
#         plt.imshow(V[t].reshape(res, res), cmap=matplotlib.cm.Greys_r)
#         # Save it
#         fname = logdir + '/' + str(t) + '.png'
#         plt.savefig(fname)


def make_bouncing_ball_dataset(data_path, res, n_ball, T, N_train, N_val):
    train_data = zeros((N_train, T, res, res))
    for i in range(N_train):
        train_data[i] = bounce_vec(res=res, n=n_ball, T=T).reshape((T, res, res))
        sys.stdout.write('\rcreating bouncing ball train clip {}/{}'.format(i, N_train))
        sys.stdout.flush()

    train_data = train_data.reshape((N_train, T, res, res))
    save(os.path.join(data_path, 'bouncing_balls_train_data.npy'), train_data)
    print()

    val_data = zeros((N_val, T, res, res))
    for i in range(N_val):
        val_data[i] = bounce_vec(res=res, n=n_ball, T=T).reshape((T, res, res))
        sys.stdout.write('\rcreating bouncing ball val clip {}/{}'.format(i, N_val))
        sys.stdout.flush()

    val_data = val_data.reshape((N_val, T, res, res))
    save(os.path.join(data_path, 'bouncing_balls_val_data.npy'), val_data)



# if __name__ == "__main__":
#     res = 30
#     T = 100
#     N = 4000
#     dat = empty((N), dtype=object)
#     for i in range(N):
#         dat[i] = bounce_vec(res=res, n=3, T=100)
#     data = {}
#     data['Data'] = dat
#     scipy.io.savemat('bouncing_balls_training_data.mat', data)
#
#     N = 200
#     dat = empty((N), dtype=object)
#     for i in range(N):
#         dat[i] = bounce_vec(res=res, n=3, T=100)
#     data = {}
#     data['Data'] = dat
#     scipy.io.savemat('bouncing_balls_testing_data.mat', data)

    # show one video
    # show_sample(dat[1])
    # ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4