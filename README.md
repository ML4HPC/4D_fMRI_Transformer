<div align="center">    

# 4D fMRI Transformer for medical prediction

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.7+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

BNL Hackathon 2022 -Team "Extreme scale spatiotemporal learning to neuroscience and genetics"

This is the implementation of 4D fMRI Transformers. This repo was built based on https://github.com/GonyRosenman/TFF.
You can test the codes by running the shell(batch) scripts in 'scripts' folder. Our model consists of 3 phases: CNN Autoencoder(Encoder+Decoder), Autoencoder(Encoder+Decoder)+Transformers, Encoder+Transformers. Also, we have trained our model with ABCD (8700 subjects) and HCP (1100 subjects).


</div>

Introduction

Spatiotemporal dynamics is the key to the brain’s adaptive functioning to the ever-changing environment. Transformer, a kind of the deep neural networks, has continued to show a great capability of learning underlying rules of temporal and sequence data across many domains including natural language processing, computer vision, sequence understanding, and bioinformatics. However, its deemed impact on neuroscience has yet to be tested rigorously. Here we test transformers as a data-driven encoder for spatiotemporal dynamics and the utility of the learning representations (embeddings) to account for an individual’s biological, cognitive, and behavioral outcomes.

Method

We built self-attention layers on Convolutional Neural Network (CNN) layers to project fMRI volumes into low dimension and learn meaningful representations for predicting the target variables. We pretrained our model with autoencoder objectives to compress spatial information and contrastive learning objectives to learn temporal relationships.  We trained and validated our model with 9,485 resting state fMRIs from the adolescent brain and cognitive development (ABCD) study and 1,084 resting state fMRIs from the Human Connectome Project (HCP). We compared the classification or regression performance of sex, age, and clinical outcomes in two datasets with previous methods. We trained, validated, and optimized several baseline models (e.g. multi layer perceptron, LSTM) with atlas based time series data–the standard type of feature widely used in human brain imaging. 

Result

In classifying an individual’s sex in HCP dataset, our brain transformer showed 0.9 accuracy and 0.95 AUROC, 15.3% increase in accuracy compared to baseline models. At the same task in the ABCD dataset, our model showed 0.79 accuracy and 0.83 AUROC, 31.6% increase in accuracy compared to baseline models. In classification and regression tasks for other variables, our model showed improved performances compared to other methods as well.

 

 

Conclusion

  Our results suggest that deep neural networks learn the representations of spatiotemporal dynamics that show such utility in predicting an individual’s biological and cognitive outcomes above and beyond the traditional functional connectivity measures. These exciting early findings present the feasibility of extreme-scale spatiotemporal computational learning in neuroscience. 
    

Our scripts were tested in Perlmutter, which supports multi-node training. By default, DataParallel without multi-node training is executed.
We recommend you to run the codes using dummy datasets, because ABCD and HCP dataset amounts to several Terabytes.

    ### DataParallel
    python main.py --step 1 --dataset_name Dummy --batch_size_phase1 64
  
    ### DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --step 1 --dataset_name Dummy --batch_size_phase1 64

