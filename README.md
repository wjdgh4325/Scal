# Synthetic data
    # Lognormal distribution
	# Cox
	    # Scal
		python train.py --name scal --dataset synthetic --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --batch_size 200 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/

	# MTLR
	    # Scal
		python train.py --name scal --dataset synthetic --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/
  
  	# Parametric(lognormal)
	    # Scal
		python train.py --name scal --dataset synthetic --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-4 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/
