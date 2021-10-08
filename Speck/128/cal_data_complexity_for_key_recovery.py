import numpy as np
import scipy.stats as st


def cal_DC(p0=0.5, p1=0.5, p2=0.5, p3=0.5, bp=0.005, bn=0.005):
    z_1_bp = st.norm.ppf(1 - bp)
    z_1_bn = st.norm.ppf(1 - bn)
    mu_p = p0 * p1 + (1 - p0) * p3
    mu_n = p0 * p2 + (1 - p0) * p3
    sig_p = np.sqrt(p0 * p1 * (1 - p1) + (1 - p0) * p3 * (1 - p3))
    sig_n = np.sqrt(p0 * p2 * (1 - p2) + (1 - p0) * p3 * (1 - p3))
    # print('z_1_bp is ', z_1_bp)
    # print('z_1_bn is ', z_1_bn)
    # print('mu_p is ', mu_p)
    # print('mu_n is ', mu_n)
    # print('sig_p is ', sig_p)
    # print('sig_n is ', sig_n)
    x = z_1_bp * sig_p + z_1_bn * sig_n
    y = np.abs(mu_p - mu_n)

    N = (x / y) * (x / y)
    dc = np.log2(N)
    print('the weight of data complexity is ', dc)

    # calculate the decision threshold t
    sig = sig_p * np.sqrt(N)
    mu = mu_p * N
    t = mu - sig * z_1_bp
    print('t is ', t)
    # print('u_p is ', mu)
    # print('sig_p is ', sig)


# c3 = 0.5, <= 2 bits  0.392551
# print('6 + 7')
# cal_DC(p0=2**(-32), p1=0.826497, p2=0.563643, p3=0.127439, bp=0.005, bn=2**(-48))

# c3 = 0.5, <= 2 bits,    22 bits  29 ~ 8
print('7 + 8')
cal_DC(p0=2**(-1), p1=0.5243686, p2=0.3782447, p3=0.2694404, bp=0.005, bn=2**(-22))
# dc = 2370, t = 881

# c3 = 0.55, <= 2 bits,    22 bits  29 ~ 8
print('7 + 8')
cal_DC(p0=2**(-1), p1=0.523396, p2=0.397973, p3=0.270115, bp=0.005, bn=2**(-5))
# dc = 1108, t = 399,   for informative bits, 20~16,  p2 = p2_{d1 = 2]

cal_DC(p0=2**(-1), p1=0.523396, p2=0.406909, p3=0.270115, bp=0.005, bn=2**(-8))
# dc = 1788, t = 657,   for informative bits, 20~13,  p2 = p2_{d1 = 3]

cal_DC(p0=2**(-1), p1=0.523396, p2=0.42206, p3=0.270115, bp=0.005, bn=2**(-10))
# dc = 2781, t = 1038,   for informative bits, 20~11  p2 = p2_{d1 = 3]

cal_DC(p0=2**(-1), p1=0.524459, p2=0.3503464, p3=0.2696358, bp=0.005, bn=2**(-17))
# dc = 1360, t = 494,   for informative bits, 29~13  p2 = p2_{d1 = 3]

# c3 = 0.55, <= 2 bits,    22 bits  19 ~ 8
print('7 + 8')
cal_DC(p0=2**(-4), p1=0.524086, p2=0.306583, p3=0.269685, bp=0.005, bn=2**(-12))
# dc = 42296,  t = 11841,  when  p2 =
# dc = 39566,  t = 11070,  when  p2 = 0.306583
