```
python test.py --train_file ../dialog_rc_data/json/Trn.json \
                   --dev_file ../dialog_rc_data/json/Dev.json \
                   --embedding_file glove.6B.100d.txt \
                   --model cnn_lstm_UA_DA --logging_to_file log.txt \
                   --save_model model.h5 --stopwords stopwords.txt
```

https://stackoverflow.com/questions/53445345/what-is-the-pytorch-alternative-for-keras-input-shape-output-shape-get-weights

https://www.tensorflow.org/guide/keras/custom_layers_and_models

https://stackoverflow.com/questions/51803437/how-do-i-use-a-saved-model-in-pytorch-to-predict-the-label-of-a-never-before-see

https://blog.csdn.net/lrt366/article/details/96211913