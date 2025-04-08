# Credit-Card-Attrition-Prediction-(Accuracy-93%)

Kaggle provides data for analyzing and predicting customer behavior, including credit card usage and churn. We will use PyTorch to implement a neural network to identify customers at risk of leaving.

Source: [https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m/data](url)

<img width="1200" src="https://nairametrics.com/wp-content/uploads/2023/10/Credit-cards-e1698396514274.png">

## Introduction

The notebook includes various exploratory data analyses (EDA) to examine relationships among factors such as Income Category, Number of Transactions, Average Utilization Ratio, Educational Level, etc. The data is then processed for ingestion into PyTorch to build a binary classification model. Details of the neural network and its implementation are documented in the notebook.

## Dependencies

Pytorch, Pandas, Numpy, Matplotlib, Seaborn, Sklearn

## Exploratory Data Analysis

![image](https://github.com/user-attachments/assets/d521f9a0-3a49-4eb3-a2b4-e31d63642551)

The dataset has an imbalanced attrition rate, with existing customers accounting for approximately 84% of the data. This imbalance should be considered when training the model. <p>
![image](https://github.com/user-attachments/assets/171002df-8236-4e53-8747-29f9efb320e1) <p>
Attrited customers tend to have lower total transaction counts compared to existing customers. The median transaction count for attrited customers is lower (41 vs 72), and their overall range of transaction counts is narrower compared to existing customers, indicating that lower engagement may be a strong predictor of attrition. <p>

![image](https://github.com/user-attachments/assets/25f6d710-e77b-4e5b-a39f-8bf47686bc5d) <p>
Customers in the "Less than $40K" income category make up the largest share of the dataset and exhibit the highest attrition rate. This suggests a potential relationship between lower income levels and a higher likelihood of churn.

## Neural Network Model Training

First, we need to one-hot encode categorical variables before feeding them into the ML model. Since the dataset is fairly simple, we can build a 3 layer neural network. <p>
```python  
def __init__(self, input_size):  
    super().__init__()
    self.layer1 = nn.Linear(input_size, 64)  # Input size: 23  
    self.relu1 = nn.ReLU()
    self.layer2 = nn.Linear(64, 32)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(0.2)
    self.layer3 = nn.Linear(32, 16)
    self.relu3 = nn.ReLU()
    self.output = nn.Linear(16, 1)
```  

The network will consist of Linear and ReLU layers for the classification tasks, with a Dropout layer for regularization to reduce overfitting. A Sigmoid layer is not necessary as we will use the BCEWithLogitsLoss loss function in the main loop; the optimizer used is Adam. After fine-tuning, a combination of a learning rate of 0.001 and a weight decay of 0.001 yields the highest and most stable training results. After running for 200 epochs, the accuracy achieved is 93.34%.

Given this model, banks can apply it to:
- Identify customer segments with high attrition risk and proactively target them with personalized retention strategies.
- Analyze loyal customer segments to discover key drivers of retention, which can inform loyalty programs and targeted offerings.
- Prioritize marketing and resource allocation by estimating the potential return on investment (ROI) from retaining different customer groups.


