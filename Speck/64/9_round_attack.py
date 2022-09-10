import numpy as np
import speck as sp
from os import urandom
from keras.models import load_model
import time

WORD_SIZE = sp.WORD_SIZE()
MASK_VAL = 2**WORD_SIZE - 1
master_key_bit_length = 96


def extract_sensitive_bits(raw_x, bits=None):
    # get new-x according to sensitive bits
    id0 = [WORD_SIZE - 1 - v for v in bits]
    id1 = [v + WORD_SIZE * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


def make_plaintext_structure(diff, neutral_bits):
    p0l = np.frombuffer(urandom(4), dtype=np.uint32)
    p0r = np.frombuffer(urandom(4), dtype=np.uint32)
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0
        d1 = 0
        for j in i:
            d = 1 << j
            d0 |= d >> WORD_SIZE
            d1 |= d & MASK_VAL
        p0l = np.concatenate([p0l, p0l ^ d0])
        p0r = np.concatenate([p0r, p0r ^ d1])
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r


def attack_with_one_nd(cts, kg_bit_num, kg_offset, sur_kg_low, net, bits, c):
    sur_kg = []
    sur_kg_socres = []
    kg_batch = 2**kg_bit_num
    c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
    n = len(c0l)
    c0l, c0r, c1l, c1r = np.tile(c0l, kg_batch), np.tile(c0r, kg_batch), np.tile(c1l, kg_batch), np.tile(c1r, kg_batch)
    kg_high = np.arange(kg_batch, dtype=np.uint32) << kg_offset
    if sur_kg_low is not None:
        for kg_low in sur_kg_low:
            kg = kg_high | kg_low; key_guess = np.repeat(kg, n)
            d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
            d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
            raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
            x = extract_sensitive_bits(raw_x, bits)
            z = net.predict(x, batch_size=10000)
            z = np.log2(z / (1 - z)); z = np.reshape(z, (kg_batch, n))
            s = np.sum(z, axis=1)
            for i in range(kg_batch):
                if s[i] > c:
                    sur_kg.append(kg[i])
                    sur_kg_socres.append(s[i])
    else:
        kg = kg_high; key_guess = np.repeat(kg, n)
        d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
        d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
        raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
        x = extract_sensitive_bits(raw_x, bits)
        z = net.predict(x, batch_size=10000)
        z = np.log2(z / (1 - z)); z = np.reshape(z, (kg_batch, n))
        s = np.sum(z, axis=1)
        for i in range(kg_batch):
            if s[i] > c:
                sur_kg.append(kg[i])
                sur_kg_socres.append(s[i])
    return sur_kg, sur_kg_socres


def output_sk_kg_diff(sk, sur_kg, kg_scores):
    for i in range(len(sur_kg)):
        print('difference between surviving kg and sk is {}, rank score is {}'.format(hex(sk ^ np.uint32(sur_kg[i])), kg_scores[i]))


def select_top_k_candidates(sur_kg, kg_scores, k=3):
    num = len(sur_kg)
    tp = kg_scores.copy()
    tp.sort(reverse=True)
    # print('the top scores are ', tp[:5])
    if num > k:
        base = tp[k]
    else:
        return sur_kg, kg_scores
    filtered_subkey = []
    filtered_score = []
    for i in range(num):
        if kg_scores[i] > base:
            filtered_subkey.append(sur_kg[i])
            filtered_score.append(kg_scores[i])
    return filtered_subkey, filtered_score


def attack_with_dual_NDs(t, nr, diffs, NBs, nds, bits, c, k):
    assert master_key_bit_length % WORD_SIZE == 0
    m = master_key_bit_length // WORD_SIZE
    assert m == 3 or m == 4
    nets = []
    for nd in nds:
        nets.append(load_model(nd))
    acc = 0
    time_consumption = np.zeros(t)
    data_consumption = np.zeros(t, dtype=np.uint32)
    for i in range(t):
        print('attack index: {}'.format(i))
        data_num = 0
        start = time.time()
        key = np.frombuffer(urandom(m * 4), dtype=np.uint32).reshape(m, 1)
        ks = sp.expand_key(key, nr)
        tk = ks[-1][0]
        
        # stage 1, guess sk[9~0], diff index is 42
        num = 0
        while True:
            if num >= 2**4:
                num = -1
                break

            p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[0], NBs[0])
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
            sur_kg_1, kg_scores_1 = attack_with_one_nd([c0l, c0r, c1l, c1r], 10, 0, None, nets[0], bits[0], c[0])

            num += 1
            data_num += 1
            if len(sur_kg_1) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                continue
            else:
                print(' ')
                print('Stage 1: ', len(sur_kg_1), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        kg_1, kg_scores_1 = select_top_k_candidates(sur_kg_1, kg_scores_1, k[0])
        # output_sk_kg_diff(tk & 0x3ff, kg_1, kg_scores_1)
        
        # stage 2, guess sk[21~10], diff index is 47
        num = 0
        while True:
            if num >= 2**4:
                num = -1
                break
            p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[1], NBs[1])
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
            sur_kg_2, kg_scores_2 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 10, kg_1, nets[1], bits[1], c[1])
            data_num += 1
            num += 1
            if len(sur_kg_2) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                continue
            else:
                print(' ')
                print('Stage 2: ', len(sur_kg_2), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        kg_2, kg_scores_2 = select_top_k_candidates(sur_kg_2, kg_scores_2, k[1])
        # output_sk_kg_diff(tk & 0x3fffff, kg_2, kg_scores_2)

        # stage 3, guess sk[31, 22], diff index is 33
        num = 0
        while True:
            if num >= 2**4:
                num = -1
                break
            p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[2], NBs[2])
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
            sur_kg_3, kg_scores_3 = attack_with_one_nd([c0l, c0r, c1l, c1r], 10, 22, kg_2, nets[2], bits[2], c[2])
            data_num += 1
            num += 1
            if len(sur_kg_3) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                continue
            else:
                print(' ')
                print('Stage 3: ', len(sur_kg_3), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        sur_kg, kg_scores = select_top_k_candidates(sur_kg_3, kg_scores_3, k[2])
        end = time.time()
        output_sk_kg_diff(tk, sur_kg, kg_scores)

        print('{} plaintext structures are generated.'.format(data_num))
        print('the time consumption is ', end - start)
        print('')
        time_consumption[i] = end - start
        data_consumption[i] = data_num
    print('average time consumption is', np.mean(time_consumption))
    print('average structure consumption is', np.mean(data_consumption))
    # print('success rate is {}'.format(acc / t))


if __name__ == '__main__':
    # (0x400, 0)
    nd1 = './saved_model/6/42_student_6_distinguisher.h5'
    # (0x8000, 0)
    nd2 = './saved_model/6/47_student_6_distinguisher.h5'
    # (0x2, 0)
    nd3 = './saved_model/6/33_student_6_distinguisher.h5'
    selected_bits_1 = [i for i in range(17, 7, -1)]
    selected_bits_2 = [i for i in range(29, 17, -1)]
    selected_bits_3 = [31, 30] + [i for i in range(7, -1, -1)]
    diff_1 = (0x48000, 0x80)
    diff_2 = (0x900000, 0x1000)
    diff_3 = (0x240, 0x40000000)
    NB_1 = [39 - i for i in range(10)]
    NB_2 = [39 - i for i in range(10)]
    NB_3 = [29 - i for i in range(10)]
    attack_with_dual_NDs(t=100, nr=9, diffs=(diff_1, diff_2, diff_3), NBs=(NB_1, NB_2, NB_3), nds=(nd1, nd2, nd3),
                         bits=(selected_bits_1, selected_bits_2, selected_bits_3), c=(10, 10, 10), k=(3, 3, 3))