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

- Show distribution of raw/preprocessed datasets via histogram/scatter plots.
- Reduce the size of raw/preprocessed datasets via the following methods:
    - An implementation of PCA with the numpy library
    - The implementation of PCA by the sklearn library
    - The implementation of an incremental PCA by the sklearn library
    - The implementation of T-SNE by the sklearn library
- Classify raw/preprocessed datasets via K-Means and show the classification results via plots/matrices

## Files structure

- config_examples: directory with some configuration files examples to use as input for the component (see How to run section for more information)
- datasets: directory containing the raw datasets that are used by the component
- output: directory containing the output results of the component. Concretely it cotains the numerical results and plots required for the second delivery of the IML subject
- src: directory containing all the code of the component
    - data: directory containing the classes to implement the loading and preprocessing of datasets
    - algorithms: directory containing the classes to implement the different algorithms
    - optimizers: directory containing the classes to optimize the different algorithms (e.g. selecting the best params or running more than one time and selecting the best results)
    - factory: directory containing the classes to connect the different algorithms/optimizers/datasets to the main file
    - auxiliary: directory containing auxiliary methods for other classes (e.g. loading of files)
    - main: script file that runs the component
    - visualize: script file with some visualization methods

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
        "balance": 0,
        "classes_to_numerical": {
            "compensated_hypothyroid": 0,
            "negative": 1,
            "primary_hypothyroid": 2,
            "secondary_hypothyroid": 3
        },
        "only_numerical": 0,
        "prepare": [{"name": "shuffle"}]
    }
  ```
    - breast: A default configuration:
    ```
  "data": {
        "name": "breast",
        "classes_to_numerical": {
            "benign": 0,
            "malignant": 1
        },
        "prepare": [{"name": "shuffle"}]
    }
  ```
    - kropt: A default configuration:
    ```
  "data": {
        "name": "kropt",
        "balance": 0,
        "prepare": [{"name": "shuffle"}]
    }
  ```
    - csv: A default configuration to load any csv file
    ```
  "data": {
        "name": "reduced",
        "path": "../path_dataframe.csv",
        "prepare": [{"name": "shuffle"}]
    }
  ```
- algorithm (optional): configuration of the algorithm to run
    - own_pca: A default configuration:
    ```
  "algorithm": {
	       "type": "factor_analysis",
        "name": "our_pca",
        "show_variance_plots": 1,
	       "params": {"n_components": 2}
    }
  ```
    - sklearn_decomposition_pca: A default configuration, all the parameters of the sklearn library can be added to the params attribute:
    ```
  "algorithm": {
	       "type": "factor_analysis",
        "name": "decomposition_pca",
	       "params": {"n_components": 2}
    }
  ```
    - sklearn_incremental_pca: A default configuration, all the parameters of the sklearn library can be added to the params attribute:
    ```
  "algorithm": {
	       "type": "factor_analysis",
        "name": "incremental_pca",
	       "params": {"n_components": 2, "batch_size": 100}
    }
  ```
    - tsne: A default configuration, all the parameters of the sklearn library can be added to the params attribute:
    ```
  "algorithm": {
	       "type": "factor_analysis",
        "name": "tsne",
	       "params": {"n_components": 2, "n_iter": 250}
    }
  ```
    - kmeans: A default configuration:
    ```
  "algorithm": {
	       "type": "unsupervised",
        "name": "kmeans",
	       "params": {"n_clusters": 2, "max_iter": 100}
    }
  ```
- optimizer (optional): configuration to optimize some algorithms
    - kmeans: A default configuration to optimize the number of clusters of the kmeans algorithm with the davies-bouldin metric
    ```
  "optimizer": {
        "name": "unsupervised",
        "metrics": ["davies_bouldin"],
        "params": [{"name": "n_clusters", "values": [2, 3, 4, 5, 6]}],
        "use_best_parameters": 0,
        "n_runs": 5
    }
  ```
- charts: configuration to generate charts
    - class_frequencies_separated: show class frequencies of the classes of the input dataset. A default configuration:
    `{"name": "class_frequencies_separate"}`
    - class_frequencies: show class frequencies of the classes of the input dataset and the predicted labels by the algorithm. A default configuration:
    `{"name": "class_frequencies"}`
    - show_metrics_table: show the specified metrics of the predicted clusters. A default configuration:
    `{"name": "show_metrics_table", "metrics": ["davies_bouldin", "adjusted_rand"]}`
    - show_classification_report: show classification report (precision, recall, accuracy...). A default configuration:
    `{"name": "show_classification_report"}`
    - show_confusion_matrix: show confusion matrix. A default configuration:
    `{"name": "show_confusion_matrix"}`
    - show_feature_histograms: show histograms for the features of the input dataset. A default configuration:
    `{"name": "show_feature_histograms", "bins": 20}`
    - show_correlation_among_variables: show correlation matrix among the features of the input dataset. A default configuration:
    `{"name": "show_correlation_among_variables","figsize": [10,20], "title": "title"}`
    - show_parallel_coordinates: show the parallel coordinates for the features of the input dataset. A default configuration:
    `{"name": "show_parallel_coordinates","figsize": [10,20], "title": "title"}`
    - show_pair_wise_scatter_plot: show the pair wise scatter plot for the features of the input dataset. A default configuration:
    `{"name": "show_pair_wise_scatter_plot", "title": "title", "original": 1, "predicted": 0}`
    - show_clusters_2d: show a representation in 2d of the input dataset. A default configuration:
    `{"name": "show_clusters_2d","title": "Title","figsize": [10,20]}`
