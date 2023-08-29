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
		python train.py --name scal --dataset synthetic --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/

    # Weibull distribution
	# Cox
	    # Scal
		python train.py --name scal --dataset synthetic --synthetic_dist weibull --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/

	# MTLR
	    # Scal
		python train.py --name scal --dataset synthetic --synthetic_dist weibull --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/
  
  	# Parametric(lognormal)
	    # Scal
		python train.py --name scal --dataset synthetic --synthetic_dist weibull --alpha 1 --beta 1 --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --num_s 1 --seed 1 --data_dir data/lognormal_54.89%/
	    # Xcal
		python train_xcal.py --name xcal --dataset synthetic --synthetic_dist weibull --batch_size 200 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --seed 1 --data_dir data/lognormal_54.89%/

# Real data
    # METABRIC
        # Cox
	    # Scal
		python train.py --name scal --dataset metabric --alpha 1 --beta 1 --batch_size 32 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1 --data_dir data/metabric/
	    # Xcal
		python train_xcal.py --name xcal --dataset metabric --batch_size 32 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --seed 1 --data_dir data/metabric/

	# MTLR
	    # Scal
		python train.py --name scal --dataset metabric --alpha 1 --beta 1 --batch_size 32 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --num_s 1 --seed 1 --data_dir data/metabric/
	    # Xcal
		python train_xcal.py --name xcal --dataset metabric --batch_size 32 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --seed 1 --data_dir data/metabric/
  
  	# Parametric(lognormal)
	    # Scal
		python train.py --name scal --dataset metabric --alpha 1 --beta 1 --batch_size 32 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --num_s 1 --seed 1 --data_dir data/metabric/
	    # Xcal
		python train_xcal.py --name xcal --dataset metabric --batch_size 32 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --seed 1 --data_dir data/metabric/

    # SUPPORT
        # Cox
	    # Scal
		python train.py --name scal --dataset support --alpha 1 --beta 1 --batch_size 128 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --num_s 1 --seed 1 --data_dir data/support/
	    # Xcal
		python train_xcal.py --name xcal --dataset support --batch_size 128 --optimizer adam --model SyntheticNN --model_dist cox --censor True --lr 1e-4 --lam 1.0 --seed 1 --data_dir data/support/

	# MTLR
	    # Scal
		python train.py --name scal --dataset support --alpha 1 --beta 1 --batch_size 128 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --num_s 1 --seed 1 --data_dir data/support/
	    # Xcal
		python train_xcal.py --name xcal --dataset support --batch_size 128 --optimizer adam --model MTLRNN --model_dist mtlr --censor True --lr 1e-3 --num_epochs 2000 --lam 1.0 --seed 1 --data_dir data/support/
  
  	# Parametric(lognormal)
	    # Scal
		python train.py --name scal --dataset support --alpha 1 --beta 1 --batch_size 128 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --num_s 1 --seed 1 --data_dir data/support/
	    # Xcal
		python train_xcal.py --name xcal --dataset support --batch_size 128 --optimizer adam --model SyntheticNN --model_dist lognormal --censor True --lr 1e-3 --num_epochs 500 --lam 1.0 --seed 1 --data_dir data/support/

  # Record the model result (example)
  	# Scal
   	    python test.py --name 0829 --dataset synthetic --batch_size 10000 --model SyntheticNN --model_dist cox --censor True --lam 1.0 --phase test --dropout_rate 0.0 --ckpt_path ckpts\cox_scal_11_20_ds_synthetic_lognormal_lam1.0_dr0.1_bs200_lr0.001_optimadam_epoch300_seed1\best.pth.tar 
	# Xcal
 	    python test_xcal.py --name 0829 --dataset synthetic --batch_size 10000 --model SyntheticNN --model_dist cox --censor True --lam 1.0 --phase test --dropout_rate 0.0 --ckpt_path ckpts\cox_xcal_20_ds_synthetic_lognormal_lam1.0_dr0.1_bs200_lr0.001_optimadam_epoch300_seed1\best.pth.tar 

	# Excel
 	    In evaluator/model_evaluator_test.py and evaluator/model_evaluator_test_xcal.py, directory of excel file to be saved should be clarified. And "tmp.xlsx" file (which will save the result) should exist in the directory
