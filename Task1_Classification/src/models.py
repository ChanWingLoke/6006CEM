from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from tensorflow import keras

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

def build_keras_model(input_dim: int, hidden_units=64, dropout=0.2, lr=1e-3):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden_units, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(hidden_units//2, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), "accuracy"]
    )
    return model

def get_keras_classifier(input_dim: int):
    # scikeras wrapper integrates with sklearn GridSearchCV
    clf = KerasClassifier(
        model=build_keras_model,
        input_dim=input_dim,
        hidden_units=64,
        dropout=0.2,
        lr=1e-3,
        epochs=25,
        batch_size=64,
        verbose=0,
    )
    return clf
