import speck as sp
import numpy as np
import heapq

from pickle import dump
from keras.models import Model, load_model
from os import urandom


net8_path = './saved_model/student/0x0-0x80/hard_label/29_8_student_8_distinguisher.h5'
MASK_VAL = 2 ** sp.WORD_SIZE() - 1


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [sp.WORD_SIZE() - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + sp.WORD_SIZE() * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]
    return new_x


def make_target_diff_samples(n=2**12, nr=10, diff=(0x2800, 0x10)):
    p0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    p0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]

    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)

    key = np.frombuffer(urandom(16), dtype=np.uint64).reshape(2, -1)
    ks = sp.expand_key(key, nr)

    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    # print('true key type is ', type(ks[nr-1][0]))
    return c0l, c0r, c1l, c1r, ks[nr-1][0] & np.uint64(0x3fffff)


def naive_key_recovery_attack(t=100, n=2**12, th=11070, nr=10, c3=0.55, net=net8_path, diff=(0x2800, 0x10), bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    cur_net = load_model(net)

    acc = 0
    cnt = np.zeros((t, 2**22), dtype=np.uint16)
    tk = np.zeros(t, dtype=np.uint64)
    for i in range(t):
        print('i is ', i)
        c0l, c0r, c1l, c1r, true_key = make_target_diff_samples(n=n, nr=nr, diff=diff)
        # print('true key is ', true_key & np.uint64(0x1ff0))
        print('true key is ', hex(true_key))
        # tk[i] = (true_key & np.uint64(0x1ff0))        # >> np.uint64(3)
        tk[i] = true_key

        bc = 0
        bk = 0      # -1不行
        for sk in range(2**22):
            # key_guess = (true_key & np.uint64(0x3fe00f)) + np.uint64(sk << 4)
            key_guess = sk
            t0l, t0r = sp.dec_one_round((c0l, c0r), key_guess)
            t1l, t1r = sp.dec_one_round((c1l, c1r), key_guess)
            raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            X = extract_sensitive_bits(raw_X, bits=bits)
            Z = cur_net.predict(X, batch_size=10000)
            Z = np.squeeze(Z)

            if key_guess == true_key:
                print('true key cnt is ', np.sum(Z > c3))

            if np.sum(Z > c3) >= bc:
                bc = np.sum(Z > c3)
                bk = key_guess
                print('difference between cur key and true key is ', hex(true_key ^ np.uint64(key_guess)))
                print('new best sk is ', bk, 'cur bc is ', bc)

            cnt[i][sk] = np.sum(Z > c3)

        print('difference between bk and true key is ', hex(bk ^ true_key))
        print('the number of surviving keys is ', np.sum(cnt[i, :] > th))
        if bk == true_key:
            acc = acc + 1
    acc = acc / t
    print('total acc is ', acc)

    return acc, cnt, tk


# acc, cnt, tk = naive_key_recovery_attack(t=100, n=5200, nr=10, c3=0.55, net=net7_path, diff=(0x2800, 0x10))
# np.save('./key_recovery_record/2_7_0.55_5200_cnt_record.npy', cnt)
# np.save('./key_recovery_record/2_7_0.55_5200_true_keys.npy', tk)

# threshold = 9701
# acc, cnt, tk = naive_key_recovery_attack(t=100, n=33310, nr=10, c3=0.55, net=net7_path, diff=(0x2800, 0x10))
# np.save('./key_recovery_record/2_7_0.55_33310_cnt_record.npy', cnt)
# np.save('./key_recovery_record/2_7_0.55_33310_true_keys.npy', tk)

# threshold = 3031923
# acc, cnt, tk = naive_key_recovery_attack(t=100, n=10797500, nr=11, c3=0.55, net=net7_path, diff=(0x211, 0xa04))
# np.save('./key_recovery_record/3_7_0.55_10797500_cnt_record.npy', cnt)
# np.save('./key_recovery_record/3_7_0.55_10797500_true_keys.npy', tk)

selected_bits = [29 -i for i in range(22)]      # 29 ~ 8
# threshold = 875
# acc, cnt, tk = naive_key_recovery_attack(t=100, n=1108, th=399, nr=11, c3=0.55, net=net8_path, diff=(0x1000, 0x10), bits=selected_bits)
acc, cnt, tk = naive_key_recovery_attack(t=100, n=2370, th=881, nr=11, c3=0.55, net=net8_path, diff=(0x1000, 0x10), bits=selected_bits)
np.save('./key_recovery_record/2_8_0.55_2370_cnt_record.npy', cnt)
np.save('./key_recovery_record/2_8_0.55_2370_true_keys.npy', tk)
# diff=(0x1000, 0x10)   p1 = 2^(-1)
# diff=(0x120200, 0x202)

