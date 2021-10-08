import teacher_net as tn
import speck as sp


sp.check_testvector()
# model_folder = './saved_model/teacher/0x0-0x80/'
# model_folder = './saved_model/teacher/0x80-0x480/'

model_folder = './saved_model/teacher/0x0-0x80/'
version=128
tn.train_speck_distinguisher(10, num_rounds=8, depth=1, diff=(0x0, 0x80), version=version, folder=model_folder)

