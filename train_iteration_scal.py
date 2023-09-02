import os

for i in range(1, 51):
    for j in [1, 10, 100, 1000]:
        for k in [1, 2, 5, 10, 20, 50]:
            print("seed:", i, "lambda:", j, "s:", k)
            file_list = "train.py --name scal --dataset synthetic --data_dir data/lognormal_54.89%/ --batch_size 200 --censor True --lam " + str(j) + " --model SyntheticNN --model_dist lognormal --num_s " + str(k) + " --num_epochs 500 --lr 1e-3 --optimizer adam --seed " + str(i)
            os.system("python " + file_list)
        
