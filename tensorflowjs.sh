#!/bin/bash
#tensorflowjs_converter save_all tfjs/ # for model.export('save_all')
#tensorflowjs_converter --input_format=tf_saved_model  save_all tfjs/ # for model.export('save_all')
#tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model save_tf tfjs # doesn't work
#
#tensorflowjs_converter --input_format=keras  save/tf_dnn.h5 tfjs/ # for .h5
#tensorflowjs_converter --input_format=keras_keras --output_format=tfjs_layers_model save/tf_dnn.keras tfjs/
tensorflowjs_converter --input_format=keras save/tf_dnn.h5 tfjs # ok finally with tf 2.15

cp tfjs/* public/

