# API App with Flask and Python

Welcome to the API App powered by Flask and Python! This application serves as a backend API that interacts with a MySQL database and provides data endpoints for a React front-end application.

## Prerequisites

Before getting started, make sure you have the following prerequisites installed on your system:

- [Node.js](https://nodejs.org/) (required for the React front-end)
- [MySQL](https://www.mysql.com/) (ensure the MySQL server is running)
- [Python](https://www.python.org/) (3.x recommended)

## Setup

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/api-app.git
    ```

2. Navigate to the project directory:

    ```bash
    cd api-app
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the database credentials:

    - Create a `.env` file in the project directory.
    - Add the following lines to the `.env` file, replacing the placeholders with your MySQL credentials:

        ```env
        DB_HOST=your_database_host
        DB_USER=your_database_user
        DB_PASSWORD=your_database_password
        DB_NAME=your_database_name
        ```

5. Install the required Node.js packages for the React front-end:

    ```bash
    cd react-front-app
    npm install
    ```

## Data and Models

All datasets are located in the `iran_stock` folder, and the pre-trained models with 100 epochs are stored in the `trained_models` directory.

## Running the Project

1. Ensure that your MySQL server is running.

2. In the project directory, run the Flask application:

    ```bash
    python -m flask run
    ```

3. Navigate to the React front-end application using the provided link (e.g., [http://localhost:3000](http://localhost:3000)).

That's it! You have successfully set up and run the API App with Flask and Python. If you encounter any issues, please check your configurations and ensure that all prerequisites are correctly installed.
