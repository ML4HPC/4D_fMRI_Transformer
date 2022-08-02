# 4D_fMRI_Transformer

BNL Hackathon 2022 -Team "Extreme scale spatiotemporal learning to neuroscience and genetics"

This is the implementation of 4D fMRI Transformers. This repo was built based on https://github.com/GonyRosenman/TFF.
You can test the codes by running the shell(batch) scripts in 'scripts' folder. Our model consists of 3 phases: CNN Autoencoder(Encoder+Decoder), Autoencoder(Encoder+Decoder)+Transformers, Encoder+Transformers. Also, we have trained our model with ABCD (8700 subjects) and HCP (1100 subjects).


Our scripts were tested in Perlmutter, which supports multi-node training. By default, DataParallel without multi-node training is executed.
We recommend you to run the codes using dummy datasets, because ABCD and HCP dataset amounts to several Terabytes.

  ### DataParallel
  python main.py --step 1 --dataset_name Dummy --batch_size_phase1 64
  
  ### DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --step 1 --dataset_name Dummy --batch_size_phase1 64

