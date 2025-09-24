import warnings
import keras.optimizers
import keras_tuner as kt
import numpy as np
import argparse
from keras import Sequential
from keras.src.layers import Flatten
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action="ignore", message="^internal gelsd")

_save_data:bool = False
_early_stopping:bool = True

#KareasTunner random search settings
_max_trials:int = 20
_executions_per_trial:int = 2
_patience:int = 10
_epochs:int=100

#Hidden Layer Neurons Values
_min_neuron_count:int = 8
_max_neuron_count:int = 512

#Hidden Layer Values
_min_layer_count:int = 1
_max_layer_count:int = 10
_default_layer_count:int = 2

_values_string:str = ("values: \n" 
         f"max_trials: {_max_trials}\n"
         f"executions_per_trial: {_executions_per_trial}\n"
         f"patience: {_patience}\n"
         f"epochs: {_epochs}\n"
         f"\n"
         f"min_neuron_count: {_min_neuron_count}\n"
         f"max_neuron_count: {_max_neuron_count}\n"
         f"\n"
         f"min_layer_count: {_min_layer_count}\n"
         f"max_layer_count: {_max_layer_count}\n"
         f"default_layer_count: {_default_layer_count}\n" )

def main():
    print("Starting proces...")
    X_train, X_test, X_valid, y_train, y_test, y_valid = set_up_data_skitlearn()

    # Creates model
    # val_mse calue mean squared error
    # overitte, overittes existing dir
    random_search_tuner = kt.RandomSearch(
        build_model, objective="val_mse", max_trials=_max_trials, executions_per_trial=_executions_per_trial , overwrite=True,
        directory="my_dir", project_name="california_housing")

    #Create callbacks
    callbacks_list = get_callbacks()

    #Trains model
    random_search_tuner.search(X_train, y_train, epochs=_epochs,  # Trains model
                               validation_data=(X_valid, y_valid), callbacks=[callbacks_list])

    best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0] #Gets best trail
    best_trial.summary()

    best_trial.metrics.get_last_value("val_mse")

    best_model = random_search_tuner.get_best_models(num_models=1)[0] #gets the best moddel
    test_loss, mse = best_model.evaluate(X_test, y_test) #Evaluates mdoel

    print("Test loss:", test_loss)
    print("Test mse:", mse)
    print("House value:", np.sqrt(mse) * 100000 , "$")

    if _save_data:
        print("saving model")
        best_model.save("best_model.keras") #saves the best model

    print(_values_string)

def get_callbacks():
    callbacks_list = []

    if _early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(patience=_patience, restore_best_weights=True)
        callbacks_list.append(early_stopping)

    if _save_data:
        checkpoint = keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            save_best_only=True,
            monitor="val_mse",
            mode="min"
        )
        callbacks_list.append(checkpoint)
    return callbacks_list


def build_model(hp):
    print("Building new model...")

    n_hidden = hp.Int("n_hidden", min_value=_min_layer_count, max_value=_max_layer_count, default=_default_layer_count) #Value calculated by search hyper parameters(hp)
    n_neurons = hp.Int("n_neurons", min_value=_min_neuron_count, max_value=_max_neuron_count)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate) #SGD Gradiant decent Aims to lowest value and adjusted learning rate

    model = Sequential()
    model.add(Flatten(input_shape=(8,)))

    for i in range(n_hidden): #adds layers based upon numbers of neurons
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=optimizer,metrics=["mse"])
    return model

def set_up_data_skitlearn():
    print("Getting data...")
    housing = fetch_california_housing()

    X = housing.data  # Features (8 numerical columns)
    y = housing.target  # Target (median house value)

    # Split into train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42) #splits training set soo we have validation


    st = StandardScaler() # Creates scaler

    X_train = st.fit_transform(X_train) #Standaerdises the data (Standerdises outliers values)
    X_valid = st.transform(X_valid)
    X_test = st.transform(X_test)


    return X_train, X_test, X_valid, y_train, y_test, y_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-S","--save-model", action="store_true", help="Saves model if flag is true | includes checkpoints")
    parser.add_argument("-e","--epochs", type=int, help=f"numbers of epochs default is {_epochs}")
    parser.add_argument("-t","--trials", type=int, help=f"numbers of trials default is {_max_trials}")
    parser.add_argument("--no-early-stoping", action="store_true", help=f"Disables early stoping | Early stopping a regularization technique in machine learning used to prevent overfitting ")

    args = parser.parse_args()

    if args.epochs is not None:
        _epochs = args.epochs;
    if args.trials is not None:
        _max_trials = args.trials;
    if args.no_early_stoping:
        _early_stopping = False;
    if args.save_model:
        _save_data = True;

    main()
