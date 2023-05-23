# Scal
  # synthetic data (adjust num_s, lam, seed)
    # lognormal
      # Scal
        python3 train.py --name scal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model GammaNN --censor True --lr 1e-4 --num_s 1 --lam 1.0 --seed 1
      # Xcal
        python3 train_xcal.py --name xcal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model GammaNN --censor True --lr 1e-4 --num_s 1 --lam 1.0 --seed 1
    # weibull
      # Scal
        python3 train.py --name scal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model GammaNN --censor True --lr 1e-4 --num_s 1 --lam 1.0 --seed 1
      # Xcal
        python3 train_xcal.py --name xcal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model GammaNN --censor True --lr 1e-4 --num_s 1 --lam 1.0 --seed 1

