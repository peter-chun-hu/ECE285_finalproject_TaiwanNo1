# Description
This is UCSD ECE285 final project.<br>
We join Kaggle [Humpback Whale Identification Challenge](https://www.kaggle.com/c/whale-categorization-playground) and try to aplly what we learn during the class.
## Database
9850 images from 4251 classes
<p align="left">
  <img width="460" src="https://kaggle2.blob.core.windows.net/competitions/kaggle/3333/media/happy-whale.jpg">
</p>

## Requirements
1. demo
   - `launch-pytorch-gpu.sh`
   - git lfs (Please use "git lfs pull" after you git clone this file)
     - `git clone https://github.com/eddietseng1129/ECE285_finalproject_TaiwanNo1.git`
     - `cd ECE285_finalproject_TaiwanNo1`
     - `git lfs pull`
2. baseline
   - `launch-pytorch-gpu.sh`
3. transfer learning
   - `launch-pytorch-gpu.sh`
4. Siamese network
   - Cannot run on UCSD cluster because of [Keras and Tensorflow version conflict](https://github.com/keras-team/keras/issues/9900) and [Jupyter Notebook loading incorrect Python kernel](https://github.com/jupyter/notebook/issues/2563)
   - To run code on local computer, you will need to install:
     - Python3
     - Keras
     - Tensorflow
     - Jupyter Notebook (optional)
   
## Code organization
<pre>
- demo.ipynb                         -- Run a demo of our code (Train for 5 epoches and predict 100 images' label)
- code/baseline.py                   -- Implementation of baseline algorithm
- code/baseline.ipynb                -- Implementation of baseline algorithm in Jupyter Notebook
- code/transfer_learning.py          -- Implementation of transfer learning
- code/transfer_learning.ipynb       -- Implementation of transfer learning in Jupyter Notebook
- code/SiameseNet.py                 -- Implementation of siamese network
- code/SiameseNet.ipynb              -- Implementation of siamese network in Jupyter Notebook
- data/train                         -- Training data from Kaggle 
- data/test                          -- Testing data from Kaggle
- data/train.csv                     -- Label of training data
- demo_data/test                     -- 100 images from test data
- demo_data/processed_image.npy      -- Processed image data
- result/baseline_prediction.csv     -- Prediction of test data using baseline
- result/siamese_prediction.csv      -- Prediction of test data using siamese net
</pre>
