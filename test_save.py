import os

for i in [1, 2, 5, 10, 20, 50]:
    for j in [1500, 2500]:
        for k in range(1, 11):
            file_list = "test.py --name 0309 --dataset synthetic --batch_size 10000 --dropout_rate 0.0 --censor True --model GammaNN --num_s 20 --lam " + str(j) + " --ckpt_path ckpts/0309_scal_0.5_" + str(i) + "_ds_synthetic_lam" + str(j) + ".0_dr0.1_bs200_lr0.0001_optimadam_epoch300_censorTrue_seed" + str(k)  + "/best.pth.tar"
            print("s: ", i, " lam: ", j, " seed: ", k)
            os.system("python " + file_list)

for i in [20]:
    for j in [1500, 2500]:
        for k in range(1, 11):
            file_list = "test_xcal.py --name 0309 --dataset synthetic --batch_size 10000 --dropout_rate 0.0 --censor True --model GammaNN --num_s 20 --lam " + str(j) + " --ckpt_path ckpts/0309_xcal_0.5_" + str(i) + "_ds_synthetic_lam" + str(j) + ".0_dr0.1_bs200_lr0.0001_optimadam_epoch300_censorTrue_seed" + str(k)  + "/best.pth.tar"
            print("s: ", i, " lam: ", j, " seed: ", k)
            os.system("python " + file_list)