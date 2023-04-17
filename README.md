# Plant Disease Detection using CNN and Transfer Learning

Farmers and plantation businesses worldwide face losses in crop yields due to plant diseases, and traditional manual techniques for detecting plant diseases are often ineffective and time-consuming. However, deep learning-based techniques, particularly Convolutional Neural Networks (CNNs), provide a possibility to automate the process and achieve better accuracy in disease detection, revolutionizing plant disease detection. Despite the potential of CNNs, researchers face challenges due to limited availability of large datasets, the wide range of appearances of plant leaves, and the computational expense of training CNNs.

To address these challenges, this study employed CNNs to detect plant diseases using three distinct datasets sourced from Kaggle and evaluated and compared the performance of three distinct CNN architectures: Resnet18, MobileNetv2, and InceptionV3. The study pre-processed all images by resizing them to a uniform size and conducted hyperparameter tuning and transfer learning to enhance the models' performance. The study's findings provide insight into the performance of three distinct CNN architectures for plant disease detection using different datasets, showing that all three models achieved reasonably good accuracy, precision, recall, and F1-score.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all dependencies.

```bash
pip install pandas
pip install numpy
pip install Matplotlib
pip install pytorch torchvision
pip install sklearn
```
