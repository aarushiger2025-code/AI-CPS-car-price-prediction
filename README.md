
#Second-Hand Car Price Prediction System

This repository contains an AI-based application system for predicting second-hand car prices.

The project was developed as part of the course:
M. Grum – Advanced AI-based Application Systems
University of Potsdam

This repository is a fork of:
https://github.com/MarcusGrum/AI-CPS

###Overview

This project focuses on predicting car prices using Artificial Neural Network (ANN) model and an Ordinary Least Squares (OLS) regression model; trained on cleaned and preprocessed car listing data scraped from the autoscout24.de.

Model performance is evaluated using standard regression metrics and visual analysis of predictions.

###Dataset

The dataset was scraped from online car listing platform "autoscout24.de" and contains structured information about used cars.

Features included:

 -brand
 -model
 -year
 -mileage
 -fuel type
 -engine power (PS)
 -price (target variable)

###Quick-Start

Go to the scenario folder first:
`cd scenario/apply_ols`

or for ANN:
`cd scenario/apply_ann`

Run containers:
`docker compose up --build`

###Author

 -Aarushi Gupta
 -Ayushi Sachan

##License

AGPL-3.0
