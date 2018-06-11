# ECE285 final project TaiwanNo1
Humpback Whale Identification Challenge from [Kaggle](https://www.kaggle.com/c/whale-categorization-playground)
## Database
9850 images from 4251 classes
![whale](https://kaggle2.blob.core.windows.net/competitions/kaggle/3333/media/happy-whale.jpg "whale example")
## Description

## Requirements
1. baseline
   - launch-pytorch-gpu.sh
2. transfer learning
   - launch-pytorch-gpu.sh
3. Siamese network
   - Cannot run on UCSD cluster because of [Keras and Tensorflow version conflict](https://github.com/keras-team/keras/issues/9900) and [Jupyter Notebook loading incorrect Python kernel](https://github.com/jupyter/notebook/issues/2563)
   - To run code on local computer, you will need to install:
     - Python3
     - Keras
     - Tensorflow
     - Jupyter Notebook (optional)
   
## Code organization
- demo.ipynb -- Run a demo of our code ( reproduce Figure 3 of our report )
- code/baseline.py -- Implementation of baseline algorithm
- code/baseline.ipynb -- Implementation of baseline algorithm in Jupyter Notebook
- code/transfer_learning.py -- Implementation of transfer learning
- code/transfer_learning.ipynb -- Implementation of transfer learning in Jupyter Notebook
- code/SiameseNet.py -- Implementation of siamese network
- code/SiameseNet.ipynb -- Implementation of siamese network in Jupyter Notebook
- data/train -- Training data from Kaggle 
- data/test -- Testing data from Kaggle
