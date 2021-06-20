data_gen.py
    bs = 64

model.py
    res101
    原始

train.py
    训练所有层

参数
    lr = 1e-4
	reduce_lr = _reducelr.StepLR(optimizer, factor=0.2, patience=10, min_lr=1e-6)
	CrossEntropLoss
