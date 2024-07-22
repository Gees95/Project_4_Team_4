# Project_4_Team_4

# Overview & possible clinical applications

This project builds a Supervised ML Image Classifier to detect pneumonia in paediatric X-ray images. In a hospital setting, the model can help highlight X-rays with pneumonia, which can be incorporated into clinical workflows to reduce human errors of missing abnormalities and highlight abnormal X-rays quickly to clinicians even before formal reports have been completed. In an outpatient setting, the ML tool can be used to find abnormal X-rays and push them up the priority for earlier reporting from the radiology team.

# Dataset

We used a [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data), which had 5,863 xray images which we used to train and test our model.

# Model & data processing
We built a Convoluted Neural Network(CNN) with 3 CNN layers and 2 Dense Layers.

```
def create_model(img_height, img_width, filter, seed_value, color_channels=1, dropout=0):
    model = models.Sequential([
        layers.Conv2D(32, (filter, filter), activation='relu',
                      input_shape=(img_height, img_width, color_channels)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout, seed=seed_value),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout, seed=seed_value),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout, seed=seed_value),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```
