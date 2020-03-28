## Project Title

SVM (Pegasos, Adagrad, One-versus-One) & CNN


### Goal and Process

The goal is to establish different SVM from scratch to classify FashionMNIST dataset.  
For binary classification, Pegasos SVM v.s SVM with Adagrad optimization are built here and compared.   
For multi-class classification, One-Versus-One SVM v.s CNN are built and analyzed.  

### Prerequisites

Put 'train.npy''train_labels.npy''test.npy''test_labels.npy' in the same folder with 'main.py', they are part of the FashionMNIST dataset.  
These files can be found here https://drive.google.com/file/d/1EX6_Hs6g6atEl9Z1HMBOhlozu8n6Yb0g/view?usp=sharing
https://drive.google.com/file/d/1A8yHSQ1ZHsOpxr1ntgGCciP9zHgv62go/view?usp=sharing
https://drive.google.com/file/d/1VUUZqyUATdv8cbZKHyBl0sgxBCOU3IVn/view?usp=sharing
https://drive.google.com/file/d/1kuQwaFMVPcQf7vDOQHibCWCOPOU6wih8/view?usp=sharing


### Data Visualization and Analysis

#### Binary Classification
For Pegasos, the best hyperparameters are T = 150, n = 800, lambda = 1;  
For Adagrad, the best hyperparameters are T = 20, n = 100, lambda = 0.1.  
Their training error graphs are shown below:  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/pegasos%20with%20T150_n800_lambda1.png"></div>  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/adagrad%20with%20T20_n100_lambda0.1.png"></div>  
Now, let's compare the best model of Pegasos v.s Adagrad. As shown above, Adagrad achieves over 0.98 test accuracy after 1 iteration, yet Pegasos takes over 3 iterations to achieve the same and its accuracy is oscillating during the process.   Hence, Adagrad outruns Pegasos either in convergence rate or stability. The reason is that input data contain plenty of sparse parameters, exactly where Adagrad applies. Besides, the learning rate of Pegasos may be too high at first (inversely proportional to iterations), so it oscillates a lot.  

#### Multi-class Classification
For One-Versus-One SVM, the final test accuracy is 0.9423. Its training error graphs of 3 SVM models, plus test error graph, are shown below:  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/svm25.png"></div>  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/svm27.png"></div>  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/svm57.png"></div>  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/ovo.png"></div>  
For CNN, the final test accuracy is 0.9067, the training and test error graph is as follows.  
<div align=center><img src="https://github.com/MianWang123/SVM-CNN/blob/master/pics/CNN%20error.png"></div>  
Here, one-versus-one SVM performs better than CNN. SVM achieves over 0.9 training accuracy after 1 iteration, yet CNN takes over 10 iterations to do the same. The reason for this phenomenon is that the input data are linearly separable, so SVMis already enough for classication, whereas CNN imports non-linearity through activation function, which is redundant.
              

### Acknowledge  
Special thanks to ESE545 course's professor and TAs
