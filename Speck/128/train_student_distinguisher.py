import student_net as tn
import speck as sp


sp.check_testvector()
# selected_bits = [37,36,35,34,33,32,31,30,    29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8]
# selected_bits = [29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8]
# selected_bits = [20,19,18,17,16,15,14,13,12,11,10,9,8]
selected_bits = [25, 24, 23, 22, 21]
# selected_bits = [19 - i for i in range(12)]
teacher = './saved_model/teacher/0x0-0x80/8_distinguisher.h5'
model_folder = './saved_model/student/0x0-0x80/'
tn.train_speck_distinguisher(10, num_rounds=8, depth=1, diff=(0x0, 0x80), bits=selected_bits, teacher=teacher, folder=model_folder)
