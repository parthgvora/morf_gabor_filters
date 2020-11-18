import sys
import numpy as np
import pandas as pd
from urllib.request import urlopen

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage

from sklearn.model_selection import train_test_split, cross_val_score

# Cross validates a model, returning cohen's kappa and SEM
def test(data_file, reps, n_trees, 
        default_transformer_class, default_transformer_kwargs):

    
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    
    kappa = np.zeros(reps)
    for i in range(reps):

        X_train, X_test, y_train, y_test, n_classes = load_data(data_file)
        default_decider_kwargs = {"classes" : np.arange(n_classes)}

        pl = ProgressiveLearner(
                default_transformer_class = default_transformer_class,
                default_transformer_kwargs = default_transformer_kwargs,
                default_voter_class = default_voter_class,
                default_voter_kwargs = default_voter_kwargs,
                default_decider_class = default_decider_class,
                default_decider_kwargs = default_decider_kwargs)
      
        pl.add_task(X_train, y_train, num_transformers=n_trees)

        y_hat = pl.predict(X_test, task_id=0)

        acc = np.sum(y_test == y_hat) / len(y_test)
        print(acc)

        chance_pred = 1 / n_classes
        kappa[i] = (acc - chance_pred) / (1 - chance_pred)

    return np.mean(kappa) * 100, (np.std(kappa) * 100) / np.sqrt(reps)

def load_data(data_file):


    if "Hill_Valley" in data_file or "tae" in data_file:
        df = pd.read_csv(data_file)
        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy()
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y) 

    if "balance" in data_file:
        df = pd.read_csv(data_file)

        X = df[df.columns[1:]].to_numpy()
        
        labels = df.B.to_list()
        y = np.zeros(len(labels))
        
        label_dict = {"B": 0, "L": 1, "R": 2}
        for i, l in enumerate(labels):
            y[i] = label_dict[l]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y) 

    if "acute" in data_file:

        df = pd.read_table(data_file, encoding='utf-16')
        df[df == "no"] = 0
        df[df == "yes"] = 1

        data = df.to_numpy()
        temps = data[:, 0]

        temperature = []
        for i in range(len(temps)):
            temp_str = temps[i]
            temp_str = temp_str.replace(",", ".")
            temperature.append(float(temp_str))

        data[:, 0] = np.array(temperature)

        X = np.array(data[:, :5], dtype=float)

        # 6 for task 1, 7 for task 2
        y = np.array(data[:, 7], dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y) 


    return X_train, X_test, y_train, y_test, len(np.unique(y))

def main():
   
    data_files = [
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Training.data",
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_with_noise_Training.data",
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data",
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data"
                 ]

    # Parameters
    max_depth = 10
    feature_combinations = 1.5
    density = 0.5

    reps = 5
    n_trees = 10

    rf_kwargs = {"kwargs" : {"max_depth" : max_depth} }
    of_kwargs = {"kwargs" : {"max_depth" : max_depth,
                             "feature_combinations" : feature_combinations,
                             "density" : density}}

    #RF_kappa, RF_err = test(data_files[-1], reps, n_trees,
    #                        default_transformer_class=TreeClassificationTransformer,
    #                        default_transformer_kwargs=rf_kwargs)

    OF_kappa, OF_err = test(data_files[-1], reps, n_trees,
                            default_transformer_class=ObliqueTreeClassificationTransformer,
                            default_transformer_kwargs=of_kwargs)
    
    #print("RF kappa, RF err:", RF_kappa, RF_err)
    print("OF kappa, OF err:", OF_kappa, OF_err)

if __name__ == "__main__":
    main()

