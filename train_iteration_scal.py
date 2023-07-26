import os

for i in [1, 10, 100, 1000]:
    for j in [1, 2, 5, 10, 20]:
        for k in range(1, 21):
            file_list = "train.py --name scal --alpha 4 --beta 4 --dataset metabric --batch_size 64 --optimizer adam --model SyntheticNN --model_dist cox --censor True --data_dir data/metabric/ --lr 1e-4 --lam " + str(i) + " --num_s " + str(j) + " --seed " + str(k)
            os.system("python " + file_list)