import os

for k in [1, 2, 5, 10]:
    for j in [1500, 2500]:
        for i in range(1, 11):
            file_list = "train.py --name 0309_scal_0.5 --dataset synthetic --batch_size 200 --optimizer adam --model GammaNN --censor True --lr 1e-4 --lam " + str(j) + " --num_s " + str(k) + " --seed " + str(i)
            os.system("python " + file_list)
