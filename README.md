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

Parent direct contains
1. Sub directory Datasets with contains cleaned imagedatasets with names as per there class.
2. Sub directory with model name contains .ipynb files with suffix of number of classes it has programmed for. modelname_'number of classes'classes.ipynb
3. Sub directory named Model_backup contains all the saved model .pt files. 

## Datasets 
We have used pre-processed versions of following datasets
1. potato-tomato-dataset- <https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset>
2. Plantvillage-dataset- <https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color>
3. Plant disease [50 classes]- <https://www.kaggle.com/datasets/fabinahian/plant-disease-50-classes>

pre-processed images are present in the datasets folder

## Training/Validation Process
We will train the model for 50 epochs with a batch size of 64 using stochastic gradient descent (SGD) optimizer with a learning rate of 0.001 and a momentum of 0.9. We will use the cross-entropy loss function and monitor the validation accuracy during training.

Loading datasets
1. Using the steps in the .ipynb file load dataset from Dataset subdirectory.
    ```python
    path = "../Datasets/dataset_20_classes/"
    ```
2. using transform function to pre-process and apply transformations on the dataset.
    ```python
    def transform(dataset):
     train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
     train_dataset=torchvision.datasets.ImageFolder(root=dataset,transform=train_transforms)
     train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
     mean,std=get_mean_std(train_loader)
     train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
     ])
     data_set=torchvision.datasets.ImageFolder(root=dataset,transform=train_transforms)
    return data_set
    ```

3. Split the dataset into 70:10:20,( Not Needed for 2 classes dataset, as split is from the source)
    ```python
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset)-train_size-val_size
    ```

4. Create train/validation loaders. 
    ```python
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    ```
5. Pass it to the epoch loop for training and validation. 