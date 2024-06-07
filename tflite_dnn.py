import tensorflow as tf

DATA_NUM = 10_000
QTY_RANGE = 10
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 32

import pandas as pd
import numpy as np

# Define the cost of each item
watermelon_cost = 100
apple_cost = 10
grape_cost = 1

import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite") # load from file
#interpreter = tf.lite.Interpreter(model_content=tflite_quant_model) # pipeline convert from keras model
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# input
import random
watermelon_qty = random.randint(1, QTY_RANGE)
apple_qty = random.randint(1, QTY_RANGE)
grape_qty = random.randint(1, QTY_RANGE)
exact = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)

# Create a numpy array with the input data
input_data = np.array([[watermelon_qty, apple_qty, grape_qty]], dtype=np.float32)

# Set the input tensor data
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"{watermelon_qty} {apple_qty} {grape_qty} = {exact}, predict:{output_data[0, 0]:.0f}")
