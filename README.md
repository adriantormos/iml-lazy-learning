# IML - Lazy learning (TODO update this)

_This component was created as a result of the IML subject in the MAI Masters degree._

## Introduction

This README is divided into the following 4 sections:
- Main funcionalities: We explain the funcionalities of the component
- Files structure: We explain the files and directories purposes that are included in this repository
- How to install: We explain how to set up the component
- How to run: We explain how to use the component

### Main functionalities

This component has 3 main funcionalities:

- Run a KNN algorithm with user defined parameters
- Run a KNN algorithm with instance reduction methods

## Files structure

- config_examples: directory with some configuration files examples to use as input for the component (see How to run section for more information)
- datasets: directory containing the raw datasets that are used by the component
- output: directory containing the output results of the component. Concretely it cotains the numerical results required for the third delivery of the IML subject
- src: directory containing all the code of the component
    - data: directory containing the classes to implement the loading and preprocessing of datasets
    - algorithms: directory containing the classes to implement the different algorithms
    - factory: directory containing the classes to connect the different algorithms/optimizers/datasets to the main file
    - auxiliary: directory containing auxiliary methods for other classes (e.g. loading of files)
    - main: script file to run an experiment
    - main: script file to run multiple sequential experiments

### How to install

- Use an environment with python3.6
- Install the libraries in the requirements.txt

### How to run

#### Running the code

It is necessary to run the main.py file with the following parameters:
- config_path: json file with all the parameters that define the experiment to run
- output_path: (optional) path defining the directory to save the experiment results

An example:
- python3 --config_path ../config_examples/own_path.json --output_path ../output

#### Configuration files

In this part we explain briefly the different parts of the configuration file. The configuration file is splitted into 4 sections:
- data: configuration of the dataset to use and the preprocessing steps
    - hypothyroid: A default configuration:
    ```
  "data": {
        "name": "hypothyroid",
    }
  ```
    - sick: A default configuration:
    ```
  "data": {
        "name": "sick",
    }
  ```
- algorithm (optional): configuration of the algorithm to run
    - knn: A default configuration without weighting:
    ```
  "algorithm": {
	       "name": "knn",
               "k": 5,
               "distance_metric": "euclidean",
	       "voting": "majority",
	       "weighting": {"name": "equal"}
    }
  ```
    - knn: A default configuration with weighting:
    ```
  "algorithm": {
	       "name": "knn",
               "k": 5,
               "distance_metric": "euclidean",
	       "voting": "majority",
	       "weighting": {"name": "weighted_relieff", "n_iterations": 5, "nearest_values": 10, "distance_metric": "euclidean"}
    }
  ```
    - mcnn: A knn default configuration with mcnn:
    ```
  "algorithm": {
	       "name": "mcnn",
               "k": 1,
               "distance_metric": "euclidean",
	       "voting": "majority",
	       "weighting": {"name": "equal"}
    }
  ```
    - menn: A knn default configuration with menn:
    ```
  "algorithm": {
	       "name": "menn",
               "k": 1,
               "distance_metric": "euclidean",
	       "voting": "majority",
	       "weighting": {"name": "equal"}
    }
  ```
    - drop3: A knn default configuration with drop3:
    ```
  "algorithm": {
	       "name": "drop3",
               "k": 1,
               "distance_metric": "euclidean",
	       "voting": "majority",
	       "weighting": {"name": "equal"}
               "mode": "drop3"
    }
  ```
