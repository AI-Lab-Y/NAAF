import numpy as np
from os import urandom


def WORD_SIZE():
    return(64)


def ALPHA():
    return(8)


def BETA():
    return(3)


MASK_VAL = 2 ** WORD_SIZE() - 1


def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


def rol(x, k):
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)


def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)


def expand_key(k, t, version=128):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))

    ver = version // WORD_SIZE()
    for i in range(t-1):
        l[i % (ver-1)], ks[i + 1] = enc_one_round((l[i % (ver-1)], ks[i]), i)
    return(ks)


def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)


def check_testvector(version=128):
    if version == 128:
        key = (0x0f0e0d0c0b0a0908, 0x0706050403020100)
        pt = (0x6c61766975716520, 0x7469206564616d20)
        ks = expand_key(key, 32, version=version)
        ct = encrypt(pt, ks)
        if ct == (0xa65d985179783265, 0x7860fedf5c570d18):
            flag = 1
        else:
            flag = 0
    elif version == 192:
        key = (0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100)
        pt = (0x7261482066656968, 0x43206f7420746e65)
        ks = expand_key(key, 33, version=version)
        ct = encrypt(pt, ks)
        if ct == (0x1be4cf3a13135566, 0xf9bc185de03c1886):
            flag = 1
        else:
            flag = 0
    elif version == 256:
        key = (0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100)
        pt = (0x65736f6874206e49, 0x202e72656e6f6f70)
        ks = expand_key(key, 34, version=version)
        ct = encrypt(pt, ks)
        if ct == (0x4109010405c0f53e, 0x4eeeb48d9c188f43):
            flag = 1
        else:
            flag = 0
    if flag == 1:
        print("Testvector verified.")
        return (True)
    else:
        print("Testvector not verified.")
        return (False)


#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
  for i in range(4 * WORD_SIZE()):
    index = i // WORD_SIZE()
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose()
  return(X)


# baseline training data generator,  speck128/X,  version = 128/192/256
def make_train_data(n, nr, diff=(0x80, 0x0), version=128):
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1
    plain0l = np.frombuffer(urandom(8*n), dtype=np.uint64)
    plain0r = np.frombuffer(urandom(8*n), dtype=np.uint64)
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(8*num_rand_samples), dtype=np.uint64)
    plain1r[Y == 0] = np.frombuffer(urandom(8*num_rand_samples), dtype=np.uint64)

    if version == 128:
        keys = np.frombuffer(urandom(16 * n), dtype=np.uint64).reshape(2, -1)
        ks = expand_key(keys, nr, version=version)
    elif version == 192:
        keys = np.frombuffer(urandom(24 * n), dtype=np.uint64).reshape(3, -1)
        ks = expand_key(keys, nr, version=version)
    else:
        keys = np.frombuffer(urandom(32 * n), dtype=np.uint64).reshape(4, -1)
        ks = expand_key(keys, nr, version=version)

    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return(X,Y)


check_testvector(version=128)
check_testvector(version=192)
check_testvector(version=256)
