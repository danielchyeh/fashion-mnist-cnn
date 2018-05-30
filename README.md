# fashion-mnist-cnn
NTU CSIE ADLxMLDS 2017 Fall Project 0

Project Kaggle Link: https://www.kaggle.com/c/hw0-fashion-mnist

Tensorflow implementation of CNN on fashion-mnist classification.

## Dataset
- Kaggle download Link: https://www.kaggle.com/c/hw0-fashion-mnist/data. 
- dataset includes training images, labels and testing images. 
- Original dataset source web: https://github.com/zalandoresearch/fashion-mnist

## Quick start
1. Download Dataset from kaggle link above and put into the folder named data.

2. Run the .py script!
```
cd fashion-mnist-cnn
python fashion-mnist-cnn.py
```
## Training
- In fashion-mnist-cnn.py, change mode = False (line 15), then do the Training, and the model will be saved into model folder.
- If resume the training, change resume = True (line 16), program will be restoring model parameters to train 

## Testing
- After training and get the model, change mode = True (line 15) to use testing data to test model
- A .csv file will be generated and upload file to kaggle website.

