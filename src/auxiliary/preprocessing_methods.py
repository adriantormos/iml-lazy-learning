import numpy as np
import pandas as pd


def recreate_dataframe(old_dataframe: pd.DataFrame, new_values: np.ndarray) -> pd.DataFrame:
    aux = np.concatenate((new_values, old_dataframe.iloc[:,-1:]), axis=1)
    return pd.DataFrame(data = aux,
                        index = [x for x in range(new_values.shape[0])],
                        columns = [x for x in range(new_values.shape[1] + 1)])


def shuffle(dataset):
    np.random.shuffle(dataset)


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def min_max_normalize(array):
   max_value = np.max(array)
   min_value = np.min(array)
   output = np.zeros(array.shape)
   if max_value != min_value:
     output = (array - min_value)/(max_value - min_value)
   return output


def one_hot_encoding(dataFrame, nominal_columns):
    for nominal_column in nominal_columns:
        dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame[nominal_column], prefix=nominal_column)], axis=1)
        dataFrame.drop([nominal_column], axis=1, inplace=True)
    return dataFrame