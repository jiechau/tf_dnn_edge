
# tf_dnn

[gitlab]
https://tf-dnn-edge-jiechau-4076f3210ae06cebcd3c305dfe2a839ff4be8448269.gitlab.io/index.html
[gitlab]
https://jiechau.gitlab.io/tf_dnn_edge/index.html
[github]
https://jiechau.github.io/tf_dnn_edge/index.html

sousrce model:
https://colab.research.google.com/drive/1tSGHG66SPwY1IPXWLW7VIWmc0JIx9eh1?usp=sharing

backup/ 目錄紀錄的是最初版本的 model.json 

由於需要靜態網頁。所以用 github io 或 gitlab io
gitlab io 在 /public 下
github io 在 /docs 下。它的  model.js 需要編輯 在 bin 前面加入 tf_dnn_js/ (i.e.: tf_dnn_js/group1-shard1of1.bin)

那個問題後來用版本解決
https://stackoverflow.com/questions/78466700/an-inputlayer-should-be-passed-either-a-batchinputshape-or-an-inputshape

# install this first
!pip install tensorflowjs
!pip install TensorFlow==2.15.0
!pip install tensorflow-decision-forests==1.8.1
# and then training model with this code
tf.keras.backend.clear_session()
# It worked for me.




