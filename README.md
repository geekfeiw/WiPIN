# WiPIN

The codes include implementation of Butterworth filter, iFFT/FFT, feature extraction and normalization, training/test, and prediction confidences computation.

LIBLINEAR is an SVM tool that outputs the model weights, i.e., *w*, with *w*, prediction confidences computed.
Official site is [here](https://www.csie.ntu.edu.tw/~cjlin/liblinear/). To install it in MATLAB via [this answer](https://stackoverflow.com/a/15559516).


Some other tools used in WiPIN:

1. Linux CSI tool, [here](https://dhalperi.github.io/linux-80211n-csitool/). If the Unix time stamps required, please check  [this modification](https://github.com/geekfeiw/wifiperson/tree/master/datacollectioncode/wifiwithtimestamp).

1. Linear Discriminant Analysis or PCA tool can be found [here](https://lvdmaaten.github.io/drtoolbox/).

2. Support Vector Regression tool, i.e., LIBSVM, can be found [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (this cannot compute prediction scores).

