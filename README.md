
# tf_dnn_edge

- 訓練一個簡單的模型：輸入 西瓜/蘋果/葡萄 的數量, 就可以計算出的總成本。
    - 先手動生成數據 一定數量的 西瓜/蘋果/葡萄 知道是多少錢。1萬筆資料。
    - 訓練模型，訓練好了之後 可以輸入 西瓜/蘋果/葡萄 數量，就能推理出總價錢。
    - 原始訓練的過程在 [tf_dnn.py](https://gitlab.com/jiechau/tf_dnn_edge/blob/main/tf_dnn.py) 或 [這裡](https://colab.research.google.com/drive/1tSGHG66SPwY1IPXWLW7VIWmc0JIx9eh1?usp=sharing)
    - 訓練好的模型存為 h5 格式 [/save/tf_dnn.h5](https://gitlab.com/jiechau/tf_dnn_edge/blob/main/save/tf_dnn.h5)

- 將訓練好的模型轉為 tensorflow.js 可以讀取的格式。Edge 端的 瀏覽器例如 chrome 就能運行.
    - 轉出的檔案是: 一個 model.json 和一些 .bin 的檔案，[這裡](https://gitlab.com/jiechau/tf_dnn_edge/-/tree/main/tfjs?ref_type=heads)。
    - 將這些檔案交給前端開發人員
    - chrome 範例如 [index.html](https://jiechau.gitlab.io/tf_dnn_edge/index.html)，( [原始檔案](https://gitlab.com/jiechau/tf_dnn_edge/-/tree/main/public?ref_type=heads) )

- 將訓練好的模型轉為 tensorflow lite 可以讀取的格式。Edge 端的 手機例如 andriod/ios 就能運行.
    - 轉出的檔案是: 一個 model.tflite，[這裡](https://gitlab.com/jiechau/tf_dnn_edge/-/tree/main/tflite?ref_type=heads)。
    - 將這些檔案交給 手機 開發人員

