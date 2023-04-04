# Fraud-Transaction-Detector
# Online Payment Fraud Detection Project

## Introduction

Online payment fraud is a growing problem that affects both consumers and merchants. Fraudulent activities such as stolen credit card information, fake accounts, and chargebacks can result in significant financial losses for businesses and individuals. In this project, we aim to develop a machine learning model that can detect fraudulent transactions and minimize their impact.

## Dataset

We used a public dataset from [Kaggle](https://www.kaggle.com) that contains credit card transactions made by consumers 

## Methodology
In this project for splitting the data we have used "train_test_split" method from "sklearn.model" module. For training the model we have used "DecisionTreeClassifier" from "sklearn.tree". Trained the model with 90% of data by using "fit()" method.

## Results

We achieved the best performance with this model. The model had an F1 score of 0.99. 

## Conclusion

Our results demonstrate the effectiveness of machine learning models in detecting online payment fraud. However, due to the class imbalance and the limited size of the dataset, our model may not generalize well to other datasets or real-world scenarios. Therefore, further research is needed to improve the robustness and scalability of the model.
