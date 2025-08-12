import dlib
print("CUDA enabled:", dlib.DLIB_USE_CUDA)
print("GPU devices:", dlib.cuda.get_num_devices())
