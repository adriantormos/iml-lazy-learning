from scipy.io.arff import loadarff
import json
import csv
import os


def load_arff(dataset_name: str):
    f = open(os.path.abspath(os.path.join('..', 'datasets', 'all', dataset_name + '.arff')))
    data, metadata = loadarff(f)
    return data, metadata


def load_json(path):
    with open(path) as file:
        data = json.load(file)
    return data


def save_json(path, data):
    path += '.json'
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)


def print_pretty_json(data):
    print(json.dumps(data, indent=4))


def save_csv(file, rows):
    with open(file + '.csv', 'w', newline='') as csvFile:
        spamWriter = csv.writer(csvFile, delimiter=',')
        for row in rows:
            spamWriter.writerow(row)