
# GravLensNet

This project implements a Deep Residual Learning approach to classify Strong Gravitational Lensing images into three categories based on their substructure. A 20-layer ResNet model was trained on a dataset of 30,000 images (split across training, validation, and test sets). The model was optimized using data augmentation, L2 regularization, and a learning rate scheduler that reduces LR when validation loss plateaus. The final trained model was evaluated using accuracy, loss curves, and ROC-AUC analysis to assess classification performance.

I used extensively used free GPU compute offered by Google Colab, hence my entire project directory is hosted on Google Drive.
## Project Structure
The project directory has 4 sub directories: 

This is the structure as in Google Drive Project Directory. Due to Github file upload limits, I haven't uploaded all the files as below on Github. I will provide a projet struture description for Github proj directory just below the one that follows.

```bash
├── notebooks
│   ├── Training_1.ipynb ---> This has the first training attempt for the model
│   ├── Training_2.ipynb ---> This has the second training attempt for the model
│   ├── Training_3.pynb ---> This has the third training attempt for the model
│   ├── data_download_visualize.ipynb ---> This notebook has the code which downloads the data from the drive link and performing some processing, augmentation and perform some visualizations.
│   └── model_check.ipynb ---> Checks if the model is working fine
├── results
│   ├── Train_1.pth ---> This stores model weights obtained after the first training attempt(corresponds to Training_1.ipynb notebook).
│   ├── Train_2.pth ---> This stores model weights obtained after the second training attempt(corresponds to Training_2.ipynb notebook).
│   ├── Train_2_lowest_val_loss.pth ---> This stores the model weights obtained after the second training attempt, these correspond to the lowest validation loss throughout the training attempt 2(corresponds to Training_2.ipynb notebook).
│   ├── Train_3_best.pth ---> This stores the best(in terms of val loss) model weights obtained after the third training attempt(corresponds to Training_3.ipynb notebook).
│   └── Train_3_final_ep.pth ---> This stores the  model weights obtained after the last epoch of third training attempt(corresponds to Training_3.ipynb notebook).
├── notebooks
│   ├── data_.py ---> This module has the code to create dataloaders for train,test,validation datasets.
│   ├── model.py ---> This module has the code to implement a custom ResNET model with 20 layers, this takes inspiration from the implemnetation of ResNET for CIFAR-10 detailed in the seminal paper on ResNETs.
│   ├── npy.py ---> This module has the code, which deals with converting loads of .npy img files to one npz files, which speedens the loading process.
│   └──train.py ---> This is a fairly regular training loop module used in multiple deep learning tasks.
├── README.md
└── LICENSE
```
Notice that I haven't uploaded the data directory(houses dataset, available on Google Drive directory).


## Results
Results post Training attempt 3 (20 epochs)are summarized in the table below:(best case)

| Metric  | Value |
| ------------- | ------------- |
| Testset accuracy(best model weights) | 89.6%  |
| AUC (Class 0= 'no')  | 0.987  |
| AUC (Class 1= 'sphere')  | 0.979  |
| AUC (Class 2= 'vort')  | 0.968 |


ROC curves are attached below. Also Train and Val loss and accuracy plots are attached.

![download](https://github.com/user-attachments/assets/f491ee89-f06e-4a61-bcd9-fc6e21d18f96)


![download](https://github.com/user-attachments/assets/d130af3c-18d4-4a63-affd-141e32d77fb3)


## Acknowledgements : 
1. The following paper was closely followed : https://arxiv.org/abs/1909.07346
