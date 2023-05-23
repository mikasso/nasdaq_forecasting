from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from time import time
from feature_engine.creation import CyclicalFeatures

TRAIN_DATE = "2019-01-01"
VAL_DATE = "2021-01-01"
FEATURES = ["Low", "Open", "High", "Close", "Volume", "Adjusted Close"]
TARGET_FEATURE = "Adjusted Close"
TARGET = "Target"
WINDOW = 60
SANITY_CHECK = False

csv_path = "data/csv/AAL.csv"
df = pd.read_csv(csv_path, index_col="Date")[FEATURES]
df.index = pd.to_datetime(df.index, format="%d-%m-%Y")


def get_x_y(df: pd.DataFrame):
    x_arr, y_arr = [], []
    for i in range(df.shape[0] - WINDOW):
        x = df.iloc[i : i + WINDOW][FEATURES].values
        x_arr.append(x)
        y = df.iloc[i + WINDOW][TARGET]
        y_arr.append(y)
    x_arr, y_arr = np.array(x_arr), np.array(y_arr).reshape(-1, 1)
    return x_arr, y_arr


df = df.diff(1)[1:]
target = (df[TARGET_FEATURE] >= 0).astype(int)
scaler = StandardScaler()
scaler = scaler.fit(df[:TRAIN_DATE][FEATURES].values)
scaled_x = scaler.transform(df[FEATURES].values)
df = pd.DataFrame(scaler.transform(df[FEATURES].values), index=df.index, columns=df.columns)
date_index = df.index.to_series()
df["Weekday"] = date_index.dt.day_of_week / 4
df["Month Sin"] = np.sin(date_index.dt.month * (2 * np.pi / 7))
df["Month Cos"] = np.cos(date_index.dt.month * (2 * np.pi / 7))
df[TARGET] = target

df_train = df[:TRAIN_DATE]
if SANITY_CHECK:
    df_train = df_train[:100]
df_val = df[TRAIN_DATE:VAL_DATE]
df_test = df[VAL_DATE:]

x_train, y_train = get_x_y(df_train)
x_val, y_val = get_x_y(df_val)
x_test, y_test = get_x_y(df_test)

print(f"Train data dimensions: {x_train.shape}, {y_train.shape}")
print(f"Validation data dimensions: {x_val.shape}, {y_val.shape}")
print(f"Test data dimensions: {x_test.shape}, {y_test.shape}")


# Let's make a list of CONSTANTS for modelling:
LAYERS = [32, 32, 32, 1]  # number of units in hidden and output layers
M_TRAIN = x_train.shape[0]  # number of training examples (2D)
M_VAL = x_val.shape[0]  # number of test examples (2D),full=X_test.shape[0]
N = x_train.shape[2]  # number of features
BATCH = 128  # batch size
EPOCH = 50  # number of epochs
LR = 5e-2  # learning rate of the gradient descent
LAMBD = 3e-2  # lambda in L2 regularizaion
DP = 0.0  # dropout rate
RDP = 0.0  # recurrent dropout rate
print(f"layers={LAYERS}, train_examples={M_TRAIN}, test_examples={M_VAL}")
print(f"batch = {BATCH}, timesteps = {WINDOW}, features = {N}, epochs = {EPOCH}")
print(f"lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}")

# Build the Model
model = Sequential()
model.add(
    LSTM(
        input_shape=(WINDOW, N),
        units=LAYERS[0],
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
    )
)
model.add(BatchNormalization())
model.add(
    LSTM(
        units=LAYERS[1],
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
    )
)
model.add(BatchNormalization())
model.add(
    LSTM(
        units=LAYERS[2],
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=False,
        return_state=False,
        stateful=False,
        unroll=False,
    )
)
model.add(BatchNormalization())
model.add(Dense(units=LAYERS[3], activation="sigmoid"))

# Compile the model with Adam optimizer
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(lr=LR))
print(model.summary())

# Define a learning rate decay method:
lr_decay = ReduceLROnPlateau(monitor="loss", patience=1, verbose=0, factor=0.5, min_lr=1e-8)
# Define Early Stopping:
early_stop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=30 if SANITY_CHECK == False else 100,
    verbose=1,
    mode="auto",
    baseline=0,
    restore_best_weights=True,
)
# Train the model.
# The dataset is small for NN - let's use test_data for validation
start = time()
History = model.fit(
    x_train,
    y_train,
    epochs=EPOCH,
    batch_size=BATCH,
    validation_split=0.0,
    validation_data=(x_train, y_train),
    shuffle=True,
    verbose=1,
    callbacks=[lr_decay, early_stop],
)
print("-" * 65)
print(f"Training was completed in {time() - start:.2f} secs")
print("-" * 65)
# Evaluate the model:
train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=BATCH, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH, verbose=0)
print("-" * 65)
print(f"train accuracy = {round(train_acc * 100, 4)}%")
print(f"test accuracy = {round(test_acc * 100, 4)}%")
print(f"test error = {round((1 - test_acc) * M_VAL)} out of {M_VAL} examples")

# Plot the loss and accuracy curves over epochs:
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
axs[0].plot(History.history["loss"], color="b", label="Training loss")
axs[0].plot(History.history["val_loss"], color="r", label="Validation loss")
axs[0].set_title("Loss curves")
axs[0].legend(loc="best", shadow=True)
axs[1].plot(History.history["accuracy"], color="b", label="Training accuracy")
axs[1].plot(History.history["val_accuracy"], color="r", label="Validation accuracy")
axs[1].set_title("Accuracy curves")
axs[1].legend(loc="best", shadow=True)
plt.show()


# target is prediction if next timestamp is increasing or decreasing
