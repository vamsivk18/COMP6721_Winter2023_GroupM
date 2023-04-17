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

## Folder Structure

```bash
.
├── Inceptionv3
│   ├── D2_Inception_v3_10classes.ipynb
│   ├── InceptionV3_20classes.ipynb
│   ├── InceptionV3_kfold_2class_dataset.ipynb
│   ├── Inception_V3_2_classes_TL_finetunning.ipynb
│   ├── Inceptionv3_2classes.ipynb
│   ├── Inceptionv3_2classes_TL_deeptunning.ipynb
│   ├── Models_backup
│   │   model files
│   ├── reports
│   │   model reports
│   ├── t-SNE_20classes.ipynb
│   └── t-SNE_2classes.ipynb
├── MobileNet
│   ├── D2_MobileNet_v2_10classes.ipynb
│   ├── MobileNetV2_20classes-TL-finetunning.ipynb
│   ├── MobileNetV2_20classes.ipynb
│   ├── MobileNetv2_2classes.ipynb
│   ├── MobileNetv2_hyperparameter_tunning_20classes.ipynb
│   ├── Models_backup
│   │   model files
│   ├── mobileNetv2_20classes_TL_Deeptunning.ipynb
│   ├── reports
│   │   model reports
│   ├── t-SNE_20classes.ipynb
│   └── t-SNE_2classes.ipynb
├── ResNet
│   ├── Model_backup
│   │  model files
│   ├── ResNet18_Hyperparameter_tunning_2_class_dataset.ipynb
│   ├── Resnet18_20classes.ipynb
│   ├── Resnet_18_2classes.ipynb
│   ├── reports
│   │   model reports
│   ├── t-SNE_20classes.ipynb
│   └── t-SNE_2classes.ipynb

```
