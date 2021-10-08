import speck as sp

import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import random

net_path = './saved_model/teacher/0x0-0x80/5_distinguisher.h5'
block_size = 128
nr = 5


# if type = 0, randomize all the bits
def make_target_bit_diffusion_data(id=0, n=10**7, nr=7, diff=(0x0, 0x80), type=1):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint64).reshape(2, -1)
    plain0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    plain0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
    plain1r[Y == 0] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
    ks = sp.expand_key(keys, nr)
    ctdata0l, ctdata0r = sp.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sp.encrypt((plain1l, plain1r), ks)

    # generate blinding values for target bits
    k0 = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    k1 = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    # randomize the distribution of the (id)th bit
    if type == 1:
        if id < 64:
            k1 = k1 & (1 << id)
            k0 = k0 & 0
        else:
            k0 = k0 & (1 << (id - 64))
            k1 = k1 & 0
    # apply blinding masks to all samples
    ctdata0l = ctdata0l ^ k0
    ctdata0r = ctdata0r ^ k1
    # convert to input data for neural networks
    X = sp.convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    return X, Y


def test_bits_sensitivity(nr=5, net_path=net_path, diff=(0x0, 0x80), type=1, folder='./bits_sensitive_res/'):
    acc = np.zeros(block_size+1)
    net = load_model(net_path)
    X, Y = sp.make_train_data(n=10**7, nr=nr, diff=diff)
    loss, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', acc[block_size])

    if type == 1:
        for i in range(block_size):
            x, y = make_target_bit_diffusion_data(id=i, n=10**6, nr=nr, diff=diff, type=type)
            loss, acc[i] = net.evaluate(x, y, batch_size=10000, verbose=0)
            print('cur bit position is ', i)
            print('the decrease of the acc is ', acc[block_size] - acc[i])

        np.save(folder + str(nr) + '_distinguisher_bit_sensitivity.npy', acc)
    else:
        x, y = make_target_bit_diffusion_data(id=-1, n=10 ** 7, nr=nr, diff=diff, type=type)
        loss, acc_random = net.evaluate(x, y, batch_size=10000, verbose=0)
        print('when we randomize all the bits, the decrease of the acc is ', acc[block_size] - acc_random)


folder = './bits_sensitivity_res/0x0-0x80/'
test_bits_sensitivity(nr=nr, net_path=net_path, diff=(0x0, 0x80), type=1, folder=folder)