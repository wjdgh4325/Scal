import os

for i in [1, 2, 5, 10, 20, 50]:
    for j in [1, 10, 100, 1000]:
        for k in range(1, 51):
            file_list = "test.py --name scal --dataset synthetic --batch_size 10000 --dropout_rate 0.0 --censor True --model SyntheticNN --model_dist lognormal --phase test --lam " + str(j) + " --ckpt_path ckpts/lognormal_scal_11_" + str(i) + "_ds_synthetic_lam" + str(j) + ".0_dr0.1_bs200_lr0.001_optimadam_epoch500_censorTrue_seed" + str(k) + "/best.pth.tar"
            print("s:", i, "lam:", j, "seed:", k)
            os.system("python " + file_list)

for j in [1, 10, 100, 1000]:
    for k in range(1, 51):
        file_list = "test_xcal.py --name xcal --dataset synthetic --batch_size 10000 --dropout_rate 0.0 --censor True --model SyntheticNN --model_dist lognormal --phase test --lam " + str(j) + " --ckpt_path ckpts/lognormal_xcal_20_ds_synthetic_lam" + str(j) + ".0_dr0.1_bs200_lr0.001_optimadam_epoch500_censorTrue_seed" + str(k) + "/best.pth.tar"
        print("lam:", j, "seed:", k)
        os.system("python " + file_list)
