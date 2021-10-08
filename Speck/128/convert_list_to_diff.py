diff_0 = 0
diff_1 = 0

# bits_0 = [3, 11, 14, 19, 25, 27, 30, 33, 36]
# bits_1 = [3, 11, 19, 25, 59]

bits_0 = [9, 17, 20]
bits_1 = [1, 9]

for i in bits_0:
    diff_0 = diff_0 + (1 << i)
for i in bits_1:
    diff_1 = diff_1 + (1 << i)
print('diff is ', (hex(diff_0), hex(diff_1)))