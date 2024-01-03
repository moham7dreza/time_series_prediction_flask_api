# API App with Flask and Python

Welcome to the API App powered by Flask and Python! This application serves as a backend API that interacts with a MySQL database and provides data endpoints for a React front-end application.

- [Prerequisites](#prerequisites)
- [Setup](#setup)
  1. [Clone the Repository]()
  2. [Navigate to the Project Directory](#2-navigate-to-the-project-directory)
  3. [Install Python Packages](#3-install-the-required-python-packages)
  4. [Set Up the MySQL Database](#4-set-up-the-mysql-database)
  5. [Set Up Database Credentials](#5-set-up-the-database-credentials)
  6. [Run Migrations](#6-run-migrations)
  7. [For the React Front-End Source Code](#7-for-the-react-front-end-source-code)
- [Data and Models](#data-and-models)
  - [Data Storage](#data-storage)
  - [Datasets Used](#datasets-used)
  - [Data Preprocessing](#data-preprocessing)
    - [Number of Time Steps (n_steps)](#number-of-time-steps-n_steps)
  - [Prices for Prediction](#prices-for-prediction)
  - [Models Trained](#models-trained)
  - [Training Details](#training-details)
  - [Evaluation Metrics](#evaluation-metrics)
- [MySQL Usage](#mysql-usage)
- [Running the Project](#running-the-project)

## Prerequisites

Before getting started, make sure you have the following prerequisites installed on your system:

- [Node.js](https://nodejs.org/) (required for the React front-end)
- [MySQL](https://www.mysql.com/) (ensure the MySQL server is running)
- [Python](https://www.python.org/) (3.x recommended)

## Setup

### 1. Clone the Repository:
    
    git clone https://github.com/moham7dreza/time_series_prediction_flask_api

### 2. Navigate to the project directory
    
    cd time_series_prediction_flask_api
    
### 3. Install the required Python packages
    
    pip install -r requirements.txt
    
### 4. Set up the MySQL Database:

- Log in to MySQL with your MySQL client:

      mysql -u your_database_user -p

- Create a new database:

  ```sql
  CREATE DATABASE your_database_name;
  ```

- Exit the MySQL shell:

  ```sql
  EXIT;
  ```
      
### 5. Set up the database credentials:

- Create a `.env` file in the project directory.
- Add the following lines to the `.env` file, replacing the placeholders with your MySQL credentials:

    ```env
    DB_HOST=your_database_host
    DB_USER=your_database_user
    DB_PASSWORD=your_database_password
    DB_NAME=your_database_name
    ```

### 6. Run Migrations:

- In the project directory, run the following commands to create and apply migrations:

      flask db init
      flask db migrate
      flask db upgrade

This will set up the necessary tables in your MySQL database.

### 7. For the React front-end source code:

 - please visit the [React GitHub Repository](https://github.com/moham7dreza/time_series_prediction_react).

## Data and Models

### Data Storage

All datasets are located in the `iran_stock` folder, and the pre-trained models with `100` epochs are stored in the `time_seies_models` directory.


### Datasets Used

The following datasets from the Iran market were used for prediction:

- `Dollar`
- `Car`
- `Home`
- `Gold`
- `Oil`

### Data Preprocessing

All datasets have been preprocessed, and the data extraction involved computing the `weekly average` of prices for prediction.

#### Number of Time Steps (n_steps)

The data is separated into sequences with a specific number of time steps (n_steps) for training the models. This separation helps in capturing temporal dependencies. The value of n_steps can be configured based on the desired context for prediction.

### Prices for Prediction

Checklist for prices available for prediction:

- `<CLOSE>`
- `<LOW>`
- `<HIGH>`
- `<OPEN>`

### Models Trained

The following models have been trained on `univariate` and `multivariate` time series data:

- Convolutional Neural Network (`CNN`)
- Long Short-Term Memory (`LSTM`)
- Bidirectional `LSTM`
- Artificial Neural Network (`ANN`)
- Bidirectional Artificial Neural Network (`Bi-ANN`)
- Gated Recurrent Unit (`GRU`)
- Bidirectional `GRU`
- Vanilla Recurrent Neural Network (`RNN`)
- Bidirectional `RNN`

### Training Details

- Data Period: `2014` to `2022`
- Train-Test Split: `80%`

### Evaluation Metrics

Use the following metrics to calculate errors:

- Mean Absolute Error (`MAE`)
- Mean Squared Error (`MSE`)
- Mean Absolute Percentage Error (`MAPE`)
- R-squared (`R2`)
- Root Mean Squared Error (`RMSE`)

### MySQL Usage

The MySQL database is used to store the `history` of prediction properties. This includes information such as selected `models`, `datasets`, `prices`, `n_steps`, and `evaluation metrics`.

## Running the Project

1. Ensure that your MySQL server is running.

2. In the project directory, run the Flask application:

    ```bash
    python -m flask run
    ```

3. Navigate to the React front-end application using the provided link (e.g., [http://localhost:3000](http://localhost:3000)).

That's it! You have successfully set up and run the API App with Flask and Python. If you encounter any issues, please check your configurations and ensure that all prerequisites are correctly installed.
