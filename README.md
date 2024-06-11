# Catalan Elections Data

This project is a data pipeline to download, clean, and transform Catalan Elections data. It is currently a work in progress.

This is part of my master's thesis in Data Science at UOC.

## Abstract

This master's thesis explores the influence of spatial context in improving electoral forecasts using machine learning models, focusing on the Catalan elections. The main objective is to demonstrate whether incorporating the electoral spatial context at the census section level can increase the accuracy of predictive models. The employed methodology models the dataset as a spatio-temporal problem, utilizing data from previous elections, including voting percentages, economic and demographic variables, as well as the average voting percentage of adjacent census sections. Various machine learning models have been tested, such as Linear Regression, decision trees, k-nearest neighbors, and XGBoost. The results prove that it is feasible to use these algorithms to predict elections, and that variables from adjacent sections significantly improve the quality of the models. The implications of this study are relevant to the field of electoral forecasting, providing valuable insights into the Catalan political dynamics and offering a powerful tool for electoral planning and strategy.

## Getting Started

To get started with this project, clone the repository and install the necessary dependencies listed in the `requirements.txt` file.

## Installation

Follow these steps to get the project set up on your local machine:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/username/project.git
```

2. Navigate to the project directory:

```bash
cd project
```

3. Create a virtual environment:

```bash
python3 -m venv venv
```

4. Activate the virtual environment:

```bash
source venv/bin/activate
```

5. Install the dependencies:

```bash
pip install -r src/requirements.txt
```

## Usage

The main entry point of the application is `app.py` in the `src` directory. You can run the application with different arguments to perform different tasks:

- `--download`: Download the Catalan Elections data
- `--clean`: Clean the downloaded data
- `--group`: Group the cleaned data
- `--transform`: Transform the grouped data
- `--train`: Train the models

For example, to download and clean the data, you can run:

```bash
python src/app.py --download --clean
```

## Project Structure

The project is structured as follows:

- `src/`: Contains the source code of the application
- `data/`: Contains the raw and processed data
- `results/`: Contains the results of the experiments
- `.vscode/`: Contains the configuration for Visual Studio Code

## Built with

- [Sodapy](https://pypi.org/project/sodapy/)
- [Scikit-Learn](https://scikit-learn.org)
- [XGBoost](https://xgboost.readthedocs.io/)

## Autor

- **Guillem Pla Bertran** - [guillempla@protonmail.com]()

## License

This project is open source. The specific licensing details are to be determined.

## Project Status

This project is currently a work in progress. Check back for updates.
