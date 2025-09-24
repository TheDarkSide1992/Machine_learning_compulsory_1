import warnings
import keras.optimizers
import keras_tuner as kt
from keras import Sequential
from keras.src.layers import Flatten
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action="ignore", message="^internal gelsd")

def main():
    print("Starting proces...")
    X_train, X_test, X_valid, y_train, y_test, y_valid = set_up_data_skitlearn()

    # Creates model
    # val_mse calue mean squared error
    # overitte, overittes existing dir
    random_search_tuner = kt.RandomSearch(
        build_model, objective="val_mse", max_trials=20, executions_per_trial=2 , overwrite=True,
        directory="my_dir", project_name="california_housing")

    #Create callbacks
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_mse",
        mode="min"
    )

    random_search_tuner.search(X_train, y_train, epochs=40,  # Trains model
                               validation_data=(X_valid, y_valid), callbacks=[early_stopping, checkpoint])

    best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0] #Gets best trail
    best_trial.summary()

    best_trial.metrics.get_last_value("val_mse")

    best_model = random_search_tuner.get_best_models(num_models=1)[0] #gets the best moddel
    test_loss, mse = best_model.evaluate(X_test, y_test) #Evaluates mdoel

    print("Test loss:", test_loss)
    print("Test mse:", mse * 100000, "$")

    print("saving model")

    best_model.save("best_california_model.keras") #saves the best model


def build_model(hp):
    print("Building new model...")

    n_hidden = hp.Int("n_hidden", min_value=1, max_value=10, default=2) #Value calculated by search hyper parameters(hp)
    n_neurons = hp.Int("n_neurons", min_value=8, max_value=512)
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
    main()
