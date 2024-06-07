
# tf_dnn

[gitlab]
https://tf-dnn-js-jiechau-f66e7b983fa1e574fb9feb9e0b11b9da50b64b1569d84.gitlab.io/index.html
[gitlab]
https://jiechau.gitlab.io/tf_dnn_js/index.html
[github]
https://jiechau.github.io/tf_dnn_js/index.html

sousrce model:
https://colab.research.google.com/drive/1tSGHG66SPwY1IPXWLW7VIWmc0JIx9eh1?usp=sharing


那個問題後來用版本解決
https://stackoverflow.com/questions/78466700/an-inputlayer-should-be-passed-either-a-batchinputshape-or-an-inputshape

# install this first
!pip install tensorflowjs
!pip install TensorFlow==2.15.0
!pip install tensorflow-decision-forests==1.8.1
# and then training model with this code
tf.keras.backend.clear_session()
# It worked for me.




