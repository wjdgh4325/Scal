# Synthetic data
	# Lognormal distribution
	# Cox
		# Scal
		python train.py --name Cox_scal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model SyntheticNN --model_dist	cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
		# Xcal
		python train_xcal.py --name Cox_xcal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model SyntheticNN --model_dist 	cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1

	# Parametric(lognormal)
		# Scal
		python train.py --name Parametric_scal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
		# Xcal
		python train_xcal.py --name Parametric_xcal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1

	# MTLR
		# Scal
		python train.py --name MTLR_scal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
		# Xcal
		python train_xcal.py --name MTLR_xcal --dataset synthetic --synthetic_dist lognormal --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1

	# Weibull distribution
	# Cox
		# Scal
		python train.py --name Cox_scal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model SyntheticNN --model_dist	cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
		# Xcal
		python train_xcal.py --name Cox_xcal --dataset synthetic --synthetic_dist weiull --batch_size 200 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1

	# Parametric(lognormal)
		# Scal
		python train.py --name Parametric_scal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
		# Xcal
		python train_xcal.py --name Parametric_xcal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1

	# MTLR
		# Scal
		python train.py --name MTLR_scal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
		# Xcal
		python train_xcal.py --name MTLR_xcal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1
