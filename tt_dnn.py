import pandas as pd
import numpy as np
import tensorflow as tf

# Define the cost of each item
watermelon_cost = 100
apple_cost = 10
grape_cost = 1

# Define constants
DATA_NUM = 10_000
QTY_RANGE = 10
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 32

# Create an empty list to store the data
data = []

# Generate 10,000 rows of data
for _ in range(DATA_NUM):
    # Generate random quantities for each item (between 1 and QTY_RANGE)
    watermelon_qty = np.random.randint(1, QTY_RANGE)
    apple_qty = np.random.randint(1, QTY_RANGE)
    grape_qty = np.random.randint(1, QTY_RANGE)
    # Calculate the total cost
    total_cost = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)
    # Append the data to the list
    data.append([watermelon_qty, apple_qty, grape_qty, total_cost])

# Create the DataFrame
df = pd.DataFrame(data, columns=['watermelon', 'apple', 'grape', 'cost'])

# Split the data into features (X) and target (y)
X = df[['watermelon', 'apple', 'grape']].values.astype(np.float32)
y = df['cost'].values.astype(np.float32)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# Define the model using TensorFlow low-level API
class MyModel(tf.Module):
    def __init__(self):
        self.w1 = tf.Variable(tf.random.normal([3, 64], dtype=tf.float32), name='w1')
        self.b1 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='b1')
        self.w2 = tf.Variable(tf.random.normal([64, 32], dtype=tf.float32), name='w2')
        self.b2 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='b2')
        self.w3 = tf.Variable(tf.random.normal([32, 1], dtype=tf.float32), name='w3')
        self.b3 = tf.Variable(tf.zeros([1], dtype=tf.float32), name='b3')

    def __call__(self, x):
        z1 = tf.matmul(x, self.w1) + self.b1
        a1 = tf.nn.relu(z1)
        z2 = tf.matmul(a1, self.w2) + self.b2
        a2 = tf.nn.relu(z2)
        z3 = tf.matmul(a2, self.w3) + self.b3
        return z3

model = MyModel()

# Define loss and optimizer
loss_fn = tf.losses.MeanSquaredError()
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# Training step function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    for features, labels in train_dataset:
        loss = train_step(features, labels)
        epoch_loss += loss

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_dataset)}")

# Save the model
tf.saved_model.save(model, "save_tt")
#tf.saved_model.save(model, "save_tt")

# Make a prediction on new data
import random
watermelon_qty = random.randint(1, QTY_RANGE)
apple_qty = random.randint(1, QTY_RANGE)
grape_qty = random.randint(1, QTY_RANGE)
exact = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)
new_data = np.array([[watermelon_qty, apple_qty, grape_qty]], dtype=np.float32)  # Example input

# Use the model for prediction
predicted_cost = model(new_data)
print(f"{watermelon_qty} {apple_qty} {grape_qty} = {exact}, predict:{predicted_cost[0][0]:.0f}")

