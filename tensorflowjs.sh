#!/bin/bash
tensorflowjs_converter --input_format=keras  save/tf_dnn.h5 tfjs/ # for .h5
tensorflowjs_converter save_all tfjs/