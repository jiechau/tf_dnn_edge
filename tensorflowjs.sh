#!/bin/bash
tensorflowjs_converter --input_format=keras  save/tf_dnn.h5 tfjs/ # for .h5
tensorflowjs_converter save_all tfjs/ # for model.export('save_all')
tensorflowjs_converter --input_format=tf_saved_model  save_all tfjs/ # for model.export('save_all')
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model save_tf tfjs # doesn't work
