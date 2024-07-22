# Project_4_Team_4

# Overview & possible clinical applications

This project builds a Supervised ML Image Classifier to detect pneumonia in paediatric X-ray images. In a hospital setting, the model can help highlight X-rays with pneumonia, which can be incorporated into clinical workflows to reduce human errors of missing abnormalities and highlight abnormal X-rays quickly to clinicians even before formal reports have been completed. In an outpatient setting, the ML tool can be used to find abnormal X-rays and push them up the priority for earlier reporting from the radiology team.

# Dataset

We used a [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data), which had 5,863 xray images which we used to train and test our model.

# Model & data processing
We built a Convoluted Neural Network(CNN) with 3 CNN layers and 2 Dense Layers.

```python
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

We changed a combination of variables to see which produced the model with the best accuracy. The variable (Hyperparameters) changed both aspects of data processing and the model.

### Data Processing Variables
- **Image Size**
  The original images were ~1000 x 800 pixels, and we scaled them to either 300 x 300 or 150 x 150 pixels.

- **Colour Mode**
  The images were converted to 'rgb' (3 Channel) or 'grayscale' (1 channel)

- **Batch Size**
  The dataset was batched in either 16, 32 or 64 images at a time to train the model.

- **Augmentation**
  We used a data augmentation layer for the images to reduce overfitting and increase the dataset. The layer flipped and rotated images.

### Model Variables

- **Epochs**
  After trialling the model, we established no marked gain in accuracy (no reduction in loss) after about 5 epochs. Hence, we varied training from 3 and 5 epochs.

- **Filter Size**
  We applied a filter size to the first CNN Layer of the model, which we changed to 3 x 3 or 7 x 7 pixels.

- **Drop Out**
  We applied a dropout layer to the model, and we varied this between 20% and 50%.

We manipulated 7 parameters to find the optimal model, which gave us 165 iterations or models. Each model's evaluation metric was saved with the parameters to a Pandas Dataframe. The dataframe was saved as a CSV file called [results.csv](output/results.csv). Below is a snapshot of the dataframe showing the best and worst models.

![Display of Dataframe of all models](output/AllModelsResultsDataframe.png)

One issue we encountered was that we could not reproduce the exact results every time we ran the code despite sedding the seed to 42. We suspected there may be additional areas in the code, such as data augmentation or pre-processing, where the seed may need to have been set. We accept this as a limitation of our work. To see this variation, additional results.csv files have been saved in the [misc folder](output/misc) for viewing. 

# Results

As seen in the dataframe, our best-performing model had an 87.7% accuracy. We replicated this model using the parameters from the dataframe and saved the model as [xray_pneumonia_model.h5](xray_pneumonia_model.h5). We further evaluated this model using ROC and a confusion matrix to set the decision-making threshold for the classifier.

### ROC
![ROC Curve](output/roc_curve.png)

The above figure shows the ROC curve for the model. Eyeballing the graph, the Top Left Corner of the curve falls around 0.8 True predictive rate (TPR). We calculated this Top Left Corner TPR and the threshold for the sigmoid image classification using numpy. The results for this are below, showing the threshold to be 0.6
```
Optimal Threshold (Top-Left Corner)
--------------------------------------------------
Optimal Threshold: 0.6094634532928467
TPR at Optimal Threshold: 0.7974358974358975
FPR at Optimal Threshold: 0.14957264957264957
```

However, in a clinical setting, we do not want to miss any pneumonias and prefer a higher TPR (also known as Sensitivity). We calculated the threshold for 95% sensitivity (TPR). The results are below:
```
Threshold for 95.0% Sensitivity
--------------------------------------------------
Threshold for TPR 0.95: 0.4181900918483734
Actual TPR at this threshold: 0.9512820512820512
FPR at this threshold: 0.49145299145299143
```

### Confusion Matrix
To better understand the thresholds, we created multiple confusion matrices at different thresholds [0.3, 0.4, 0.5, 0.6, 0.7], which can be found in the [output folder](output) 





  
