import numpy as np
import speck as sp
from os import urandom
from keras.models import load_model
import time
from copy import deepcopy

word_size = sp.WORD_SIZE()
MASK_VAL = 2 ** word_size - 1


def extract_sensitive_bits(raw_x, bits=None):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    id1 = [v + word_size * i for i in range(4) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


# make a plaintext structure from a random plaintext pair
# diff: difference of the plaintext pair
# neutral_bits is used to form the plaintext structure
def make_plaintext_structure(diff=(0x211, 0xa04), neutral_bits=None):
    p0l = np.frombuffer(urandom(8), dtype=np.uint64)
    p0r = np.frombuffer(urandom(8), dtype=np.uint64)
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0
        d1 = 0
        for j in i:
            d = 1 << j
            d0 |= d >> word_size
            d1 |= d & MASK_VAL
        p0l = np.concatenate([p0l, p0l ^ d0])
        p0r = np.concatenate([p0r, p0r ^ d1])
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


def attack_with_one_nd(cts, kg_bit_num=None, kg_offset=None, sur_kg_low=None, nd=None, bits=None, c=None):
    c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
    n = len(c0l)
    kg_guess_num = 2**kg_bit_num

    # If there is not enough memory, please erase this line of code.
    c0l, c0r, c1l, c1r = np.tile(c0l, kg_guess_num), np.tile(c0r, kg_guess_num), np.tile(c1l, kg_guess_num), np.tile(c1r, kg_guess_num)

    sur_kg = []
    sur_kg_scores = []
    kg_high = np.arange(kg_guess_num, dtype=np.uint64) << kg_offset
    if sur_kg_low is not None:
        
        for kg_low in sur_kg_low:
            kg = kg_high | kg_low; key_guess = np.repeat(kg, n)
            d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
            d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
            raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
            x = extract_sensitive_bits(raw_x, bits)
            z = nd.predict(x, batch_size=10000)
            z = np.log2(z / (1 - z)); z = np.reshape(z, (kg_guess_num, n))
            s = np.sum(z, axis=1)
            for i in range(kg_guess_num):
                if s[i] > c:
                    sur_kg.append(kg[i])
                    sur_kg_scores.append(s[i])

        # If there is not enough memory, run this block of code instead.
        '''
        kg_low_num = len(sur_kg_low)
        for i in range(kg_low_num):
            kg_low = sur_kg_low[i]
            for kg_high in range(kg_guess_num):
                kg = (kg_high << kg_offset) + kg_low
                d0l, d0r = sp.dec_one_round((c0l, c0r), kg)
                d1l, d1r = sp.dec_one_round((c1l, c1r), kg)
                raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
                x = extract_sensitive_bits(raw_x, bits)
                z = nd.predict(x, batch_size=n)
                s = np.sum(np.log2(z / (1 - z)))
                if s > c:
                    sur_kg.append(kg)
                    sur_kg_scores.append(s)
        '''
    else:
        kg = kg_high; key_guess = np.repeat(kg, n)
        d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
        d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
        raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
        x = extract_sensitive_bits(raw_x, bits)
        z = nd.predict(x, batch_size=10000)
        z = np.log2(z / (1 - z)); z = np.reshape(z, (kg_guess_num, n))
        s = np.sum(z, axis=1)
        for i in range(kg_guess_num):
            if s[i] > c:
                sur_kg.append(kg[i])
                sur_kg_scores.append(s[i])
            
        # If there is not enough memory, run this basic version of code instead.
        '''
        for kg_high in range(kg_guess_num):
            kg = (kg_high << kg_offset)
            d0l, d0r = sp.dec_one_round((c0l, c0r), kg)
            d1l, d1r = sp.dec_one_round((c1l, c1r), kg)
            raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
            x = extract_sensitive_bits(raw_x, bits)
            z = nd.predict(x, batch_size=n)
            s = np.sum(np.log2(z / (1 - z)))
            if s > c:
                sur_kg.append(kg)
                sur_kg_scores.append(s)
        '''
    return sur_kg, sur_kg_scores


def select_top_k_candidates(sur_kg, kg_scores, k=3):
    num = len(sur_kg)
    tp = deepcopy(kg_scores)
    tp.sort(reverse=True)
    if num > k:
        base = tp[k]
    else:
        base = tp[num-1]
    filtered_subkey = []
    for i in range(num):
        if kg_scores[i] > base:
            filtered_subkey.append(sur_kg[i])
    return filtered_subkey


def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r


def attack_with_dual_NDs(t=100, nr=None, diffs=None, NBs=None, nds=None, bits=None, c=None):
    nd_num = len(nds)
    nd = []
    for i in range(nd_num):
        nd.append(load_model(nds[i]))

    for i in range(t):
        data_num = 0
        start = time.time()

        print('Test: {}'.format(i))
        key = np.frombuffer(urandom(16), dtype=np.uint64).reshape(2, -1)
        ks = sp.expand_key(key, nr)
        # stage 1, guess sk[11~0], diff index 53
        num = 0
        while 1:
            if num >= 2**4:
                num = -1
                break

            p0l_1, p0r_1, p1l_1, p1r_1 = make_plaintext_structure(diff=diffs[0], neutral_bits=NBs[0])
            c0l_1, c0r_1, c1l_1, c1r_1 = collect_ciphertext_structure(p0l_1, p0r_1, p1l_1, p1r_1, ks)
            sur_kg_1, kg_scores_1 = attack_with_one_nd([c0l_1, c0r_1, c1l_1, c1r_1], kg_bit_num=12, kg_offset=0,
                                                       sur_kg_low=None, nd=nd[0], bits=bits[0], c=c[0])
            num += 1
            # count data complexity
            data_num += 1

            print('\r {} plaintext structures generated'.format(num), end='')
            if len(sur_kg_1) == 0:
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
        # select the best candidate of the first stage
        kg_1 = select_top_k_candidates(sur_kg_1, kg_scores_1)

        # stage 2, guess sk[23~12], diff index 65
        num = 0
        while 1:
            if num >= 2**4:
                num = -1
                break

            p0l_2, p0r_2, p1l_2, p1r_2 = make_plaintext_structure(diff=diffs[1], neutral_bits=NBs[1])
            c0l_2, c0r_2, c1l_2, c1r_2 = collect_ciphertext_structure(p0l_2, p0r_2, p1l_2, p1r_2, ks)
            num += 1
            # count data complexity
            data_num += 1

            sur_kg_2, kg_scores_2 = attack_with_one_nd([c0l_2, c0r_2, c1l_2, c1r_2], kg_bit_num=12, kg_offset=12,
                                                       sur_kg_low=kg_1, nd=nd[1], bits=bits[1], c=c[1])

            print('\r {} plaintext structures generated'.format(num), end='')
            if len(sur_kg_2) == 0:
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
        # select the best candidate of the second stage
        kg_2 = select_top_k_candidates(sur_kg_2, kg_scores_2)

        # stage 3, guess sk[35~24], diff index 77
        num = 0
        while 1:
            if num >= 2**4:
                num = -1
                break

            p0l_3, p0r_3, p1l_3, p1r_3 = make_plaintext_structure(diff=diffs[2], neutral_bits=NBs[2])
            c0l_3, c0r_3, c1l_3, c1r_3 = collect_ciphertext_structure(p0l_3, p0r_3, p1l_3, p1r_3, ks)

            num = num + 1
            # count data complexity
            data_num += 1

            sur_kg_3, kg_scores_3 = attack_with_one_nd([c0l_3, c0r_3, c1l_3, c1r_3], kg_bit_num=12, kg_offset=24,
                                                       sur_kg_low=kg_2, nd=nd[2], bits=bits[2], c=c[2])

            print('\r {} plaintext structures generated'.format(num), end='')
            if len(sur_kg_3) == 0:
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

        # select the best candidate of the second stage
        kg_3 = select_top_k_candidates(sur_kg_3, kg_scores_3)

        # stage 4, guess sk[47~36], diff index 89
        num = 0
        while 1:
            if num > 2**4:
                num = -1
                break

            p0l_4, p0r_4, p1l_4, p1r_4 = make_plaintext_structure(diff=diffs[3], neutral_bits=NBs[3])
            c0l_4, c0r_4, c1l_4, c1r_4 = collect_ciphertext_structure(p0l_4, p0r_4, p1l_4, p1r_4, ks)

            num = num + 1
            # count data complexity
            data_num += 1

            sur_kg_4, kg_scores_4 = attack_with_one_nd([c0l_4, c0r_4, c1l_4, c1r_4], kg_bit_num=12, kg_offset=36,
                                                       sur_kg_low=kg_3, nd=nd[3], bits=bits[3], c=c[3])

            print('\r {} plaintext structures generated'.format(num), end='')
            if len(sur_kg_4) == 0:
                continue
            else:
                print(' ')
                print('Stage 4: ', len(sur_kg_4), ' subkeys survive')
                break

        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue

        # select the best candidate of the first stage
        kg_4 = select_top_k_candidates(sur_kg_4, kg_scores_4)

        sur_kg = kg_4
        kg_scores = kg_scores_4
        end = time.time()

        print('the final surviving subkey guesses are ', sur_kg)
        # compare returned keys and true subkey
        sk = ks[nr - 1][0]
        num = len(sur_kg)
        for j in range(num):
            print('difference between surviving kg and sk is ', hex(np.uint64(sk) ^ np.uint64(sur_kg[j])),
                  ' the rank score is ', kg_scores[j])
        print('{} plaintext structures are generated.'.format(data_num))
        print('the time consumption is ', end - start)


if __name__ == '__main__':
    nd_1 = './saved_model/7/53_student_distinguisher.h5'
    nd_2 = './saved_model/7/65_student_distinguisher.h5'
    nd_3 = './saved_model/7/77_student_distinguisher.h5'
    nd_4 = './saved_model/7/89_student_distinguisher.h5'

    bits1 = [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]
    bits2 = [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    bits3 = [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32]
    bits4 = [47, 46, 45, 44, 7,  6,  5,  4,  3,  2,  1,  0]

    diff_1 = (0x2400, 0x4)
    diff_2 = (0x2400000, 0x4000)
    diff_3 = (0x2400000000, 0x4000000)
    diff_4 = (0x400000000002, 0x4000000000)

    NBs1 = [25 - i for i in range(10)]  # 25 ~ 16
    NBs2 = [37 - i for i in range(10)]  # 37 ~ 28
    NBs3 = [49 - i for i in range(10)]  # 49 ~ 40
    NBs4 = [61 - i for i in range(10)]  # 61 ~ 52

    diffs = [diff_1, diff_2, diff_3, diff_4]
    nds = [nd_1, nd_2, nd_3, nd_4]
    bits = [bits1, bits2, bits3, bits4]
    c = [10, 10, 10, 10, 10]
    NBs = [NBs1, NBs2, NBs3, NBs4]

    attack_with_dual_NDs(t=100, nr=10, diffs=diffs, NBs=NBs, nds=nds, bits=bits, c=c)

