
# tf_dnn

- 訓練一個簡單的模型：輸入 西瓜/蘋果/葡萄 的數量, 就可以計算出的總成本。
    - 先手動生成數據 一定數量的 西瓜/蘋果/葡萄 知道是多少錢。1萬筆資料。
    - 訓練模型，訓練好了之後 可以輸入 西瓜/蘋果/葡萄 數量，就能推理出總價錢。
    - 原始模型和訓練的過程在[這裡](https://colab.research.google.com/drive/1tSGHG66SPwY1IPXWLW7VIWmc0JIx9eh1?usp=sharing)

- 將訓練好的模型轉為 tensorflow.js 可以讀取的格式。Edge 端的 瀏覽器例如 chrome 就能運行.
    - 轉出的檔案是: 一個 model.json 和一些 .bin 的檔案。
    - 將這些檔案交給前端開發人員 就能寫出 index.html
    - 範例如同這個 [tensorflow.js](https://jiechau.gitlab.io/tf_dnn_js/index.html)

- 將訓練好的模型轉為 tensorflow lite 可以讀取的格式。Edge 端的 手機例如 andriod/ios 就能運行.
    - 轉出的檔案是: 一個 model.tflite
    - 將這些檔案交給 手機 開發人員

