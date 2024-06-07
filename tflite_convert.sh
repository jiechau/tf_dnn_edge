#!/bin/bash
tflite_convert \
  --keras_model_file=save/tf_dnn.h5 \
  --output_file=tflite/model.tflite
