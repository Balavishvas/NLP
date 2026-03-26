# Install keras-tuner if not already installed:
# pip install keras-tuner

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np

np.random.seed(42)

# Generate synthetic data
age = np.random.randint(22, 58, 1200)
salary = np.random.randint(25000, 150000, 1200)
department = np.random.randint(0, 6, 1200)

performance_score = (age / 65) + (salary / 150000) + np.random.normal(0, 0.15, 1200)
promotion_eligibility = (performance_score > 0.85).astype(int)

# Train/test split
train_size = 900
X_train = [age[:train_size], salary[:train_size], department[:train_size]]
Y_train = [performance_score[:train_size], promotion_eligibility[:train_size]]
X_test = [age[train_size:], salary[train_size:], department[train_size:]]
Y_test = [performance_score[train_size:], promotion_eligibility[train_size:]]


def build_model(hp):
    input1 = Input(shape=(1,), name="Age")
    input2 = Input(shape=(1,), name="Salary")
    input3 = Input(shape=(1,), name="Department")

    embedding_dim = hp.Int("embedding_dim", min_value=3, max_value=12, step=3)
    embedding = Embedding(input_dim=6, output_dim=embedding_dim)(input3)
    flattened = Flatten()(embedding)

    merged = Concatenate()([input1, input2, flattened])

    num_layers = hp.Int("num_layers", min_value=1, max_value=4)
    x = merged
    for i in range(num_layers):
        x = Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=160, step=32),
            activation=hp.Choice("activation", ["relu", "tanh", "elu"])
        )(x)
        x = Dropout(0.2)(x)

    output1 = Dense(1, activation="linear", name="Performance_Score")(x)
    output2 = Dense(1, activation="sigmoid", name="Promotion_Eligibility")(x)

    model = Model(inputs=[input1, input2, input3], outputs=[output1, output2])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", 1e-4, 5e-3, sampling="LOG")
        ),
        loss={
            "Performance_Score": "mse",
            "Promotion_Eligibility": "binary_crossentropy"
        },
        metrics={
            "Performance_Score": "mae",
            "Promotion_Eligibility": "accuracy"
        }
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=8,
    executions_per_trial=1,
    directory="tuner_results_modified",
    project_name="mimo_model_tuning_modified"
)

tuner.search(
    X_train, Y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, Y_test)
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print("\nBest Hyperparameters:")
print(f"Embedding Dim: {best_hps.get('embedding_dim')}")
print(f"Num Hidden Layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get("num_layers")):
    print(f"Units in Layer {i + 1}: {best_hps.get(f'units_{i}')}")
print(f"Activation: {best_hps.get('activation')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

best_model.evaluate(X_test, Y_test)
predictions = best_model.predict(X_test)

print("\nSample Predicted Performance Scores:")
print(predictions[0][:5])
print("\nSample Predicted Promotion Eligibility:")
print(predictions[1][:5])
