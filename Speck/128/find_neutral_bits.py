import numpy as np
from os import urandom
import speck as sp


block_size = sp.WORD_SIZE()


def cal_neutrality(n=2**20, diff_in=(0x124a084808, 0x800000002080808), diff_out=(0x120200, 0x202), nr=2):
    neutrality = np.zeros(block_size*2, dtype=np.float)
    print('the input difference is ', (hex(diff_in[0]), hex(diff_in[1])))
    print('the output difference is ', (hex(diff_out[0]), hex(diff_out[1])))
    print('the number of encryption round is ', nr)

    num = 0
    for i in range(n):
        p0l = np.frombuffer(urandom(8), dtype=np.uint64)
        p0r = np.frombuffer(urandom(8), dtype=np.uint64)
        p1l = p0l ^ diff_in[0]
        p1r = p0r ^ diff_in[1]
        # the key schedule does not influence the difference propagation
        keys = np.frombuffer(urandom(16), dtype=np.uint64).reshape(2, -1)
        ks = sp.expand_key(keys, nr, version=128)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)
        c0l, c0r, c1l, c1r = np.squeeze(c0l), np.squeeze(c0r), np.squeeze(c1l), np.squeeze(c1r)
        if c0l ^ c1l == diff_out[0] and c0r ^ c1r == diff_out[1]:
            num = num + 1
            print('num is ', num)
            for j in range(block_size):
                t0l, t0r = p0l, p0r ^ (1 << j)
                t1l, t1r = p1l, p1r ^ (1 << j)
                d0l, d0r = sp.encrypt((t0l, t0r), ks)
                d1l, d1r = sp.encrypt((t1l, t1r), ks)
                if d0l ^ d1l == diff_out[0] and d0r ^ d1r == diff_out[1]:
                    neutrality[j] = neutrality[j] + 1
            for j in range(block_size):
                t0l, t0r = p0l ^ (1 << j), p0r
                t1l, t1r = p1l ^ (1 << j), p1r
                d0l, d0r = sp.encrypt((t0l, t0r), ks)
                d1l, d1r = sp.encrypt((t1l, t1r), ks)
                d0l, d0r, d1l, d1r = np.squeeze(d0l), np.squeeze(d0r), np.squeeze(d1l), np.squeeze(d1r)
                if d0l ^ d1l == diff_out[0] and d1l ^ d1r == diff_out[1]:
                    neutrality[j+block_size] = neutrality[j+block_size] + 1
    neutrality = neutrality / num
    print('the neutrality test results are ')
    for j in range(block_size*2):
        print('the bit index is ', j, ' the neutrality is ', neutrality[j])


# prob = 2**(-17)
cal_neutrality(n=2**25, diff_in=(0x124a084808, 0x800000002080808), diff_out=(0x1000, 0x10), nr=3)

# prob = 2**(-14)
# cal_neutrality(n=2**22, diff_in=(0x124a084808, 0x800000002080808), diff_out=(0x120200, 0x202), nr=2)

# prob = 2**(-8)
# cal_neutrality(n=2**16, diff_in=(0x10420040, 0x4000000000024000), diff_out=(0x1000, 0x10), nr=2)

# prob = 2**(-9)
# cal_neutrality(n=2**17, diff_in=(0x10420040, 0x4000000000024000), diff_out=(0, 0x80), nr=3)
