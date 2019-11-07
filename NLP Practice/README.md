### NLP with Random Forest, Multichannel CNN, and GRU
Problem Description: I am building a text classification model. The dataset needs to be formatted to get to a workable state. 
Datasets: Below is the description of the two datasets you are being provided.

* _**train.txt-**_ There are three columns in this dataset. The example below is how to read them:
Let's say our corpus had the following 2 comments:

  XYZ is the best workplace
  
  XYZ is the number one company
  
  This is how it will be structured in the train.txt dataset


  0,1,XYZ

  0,2,is

  0,3,the
  
  0,4,best
  
  0,5,workplace
  
  1,1,XYZ
  
  1,2,is
  
  1,3,the
  
  1,4,number
  
  1,5,one
  
  1,6,company
  
  There should be a total of 5572 comments in the text file. We have encrypted the words so they might not make any sense when you read them, you don’t need to worry about that during model building. 

* _**Labels_Candidate.csv-**_ Each comment is labelled as either 0 or 1. This csv file has the labels attached to the first 4000 comments and the rest are left blank. 

* _**Evaluation:**_ The provided codes build a model based on the 4000 labeled comments and provide us the predicted label for the unlabeled comments in the Label_Candidate.csv file.

# Solution

I have approached the sentence classification problem with three different methods, one based on the classical bag of words (BOW) approach combined with a random forests (RF) classifier and others based on deep learning with multiscale convolutional neural networks (CNNs) and GRU. Below, I briefly describe each approach. The python codes for each approach are attached and may be run from the Jupyter Notebook as requested to train the model and save the results.
Labels	Count	Percentage
0	3,464	87%
1	535	13%

This problem is dealing with unbalanced data set Label 0 -> 87% and label 1 -> 13%. Therefore, to balance the data and avoid giving high weight to the biased classes I take all label 1s and randomly select a one-third of label 0s each time(undersampling) (3 different datasets will be generated) to generate the models and finally take the majority vote out of three models. In this case, models have a lower possibility of overfitting. There exist comment IDs in the label file that do not exist in the training file, so I removed the labels with empty comments.  
1.  Sentence classification with Random Forests (RF)
As shown in figure 1, the pipeline consisted of importing all comments from train.txt. The raw texts were joined together to form real sentences. Since the given text is already encrypted there are two options: 
1-	Skip the preprocessing stage.
2-	(Chosen option) As you mentioned you are dealing with messy data, I assume that we should take the encrypted words as real words. Therefore, we have to clean the comments (removing non-alphabetises). There are no empty sentences in the data set.
The sentences were cleaned to remove all symbols and punctuations. The sentences have their own label, 0 or 1.

Labels | Count | Percentage
-------|-------|------------
   0   | 3,464 |    87%    
   1   | 535   |    13%

![pipe](https://user-images.githubusercontent.com/14133335/68421874-18f51780-016d-11ea-80b0-789e5328ee40.png)

    Figure 1. The pipeline for sentence classification with RF
In order to construct a BOW model based on the word counts, I used the CountVectorizer class implemented in scikit-learn. To down-weight frequently occurring words that are less informative for the classification I employed the term frequency-inverse document frequency (tf-idf) method. To this end, I used the TfidfTransformer class from scikit-learn which takes the raw term frequencies from the CountVectorizer class as input and transforms them into tf-idfs. The transformed samples were then split into training and test datasets. I held out 20% of the samples as a test dataset. These samples were not used in training the RF classifier.

 
Figure 2. Characterization of AUC vs the number of trees. The model performance saturates at 100 trees.
I used the RandomForestClassifier from scikit-learn to train the RF models. As shown in figure 2, I explored four different tree numbers for all tree training portions and noticed that the model performance plateaued at 100 trees. I selected this tree number for model evaluation. I run the models on three undersampled datasets and predict the final label by voting. The RF model achieves an area under the curve (AUC) of 0.98 (Figure 3) and F1 of 0.98 (evaluated on 1/3 label 0s and all label 1s).
 
Figure 3. ROC curve for the RF classifier with 100 trees on the test data that was not seen by the model


2.  Sentence classification with multiscale CNNs 
Similarly to the RF model, the pipeline for the CNN model started with importing, extracting and cleaning the sentences. The labeling of the sentences for each author was also identical to the one in the previous method. Here, instead of using the BOW and tf-idf to transform sentences into feature vectors, I used the built-in tensorflow.keras utility functions to tokenize, pad and embed the sentences. This will in effect transform each sentence into a vector of vectors or a n×m matrix where n is the max length of the sentences and m is the embedding size. 


Figure 4. The pipeline for sentence classification with multiscale CNNs

I employed a three-channel multiscale CNN approach where each channel has a different kernel size in order to capture different length sentence sub-structures: 4-grams, 6-grams and 8-grams. Each channel consists of 32 filters and a MaxPool layer. The output of the MaxPool is flattened and concatenated across the three channels. The concatenated outputs of the tree channels is then fed to a dense classifier which outputs the class probabilities for the sentences. 
I trained the model on virtual machine with 12 CPUs and 24 GBs of RAM. I trained the model for 10 epochs with a batch size of 16. I tuned the hyperparameters briefly using the validation dataset and performed a final evaluation on the same 20% held out test dataset as the RF model.


 
Figure 5. ROC curve for the multiscale CNN classifier on the test data that was not seen by the model
The ROC curve for the multiscale CNN model evaluated on the test dataset is shown in figure 5. The area under the curve is 0.81 which is lower than the RF model. 

In this practice I did not implement hyper parameter tunning or perform k-fold cross-validation model evaluation which all can result in a more robust predictive model.




3.  Sentence classification with multiscale GRU
I performed similar steps for fitting a GRU model. The LSTM model has one GRU and a Dense layers.


I trained the model for 10 epochs with a batch size of 16. I tuned the hyperparameters briefly using the validation dataset and performed a final evaluation on the same 20% held out test dataset as the RF model. The Accuracy, F1-measure, Precision and Recall also calculated. However, the accuracy is ~ 14% which is really low and the model has to be refined. I did not tune the GRU model. Figure 6 shows the summary of GRU model.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_6 (Embedding)      (None, 190, 100)          616800    
_________________________________________________________________
gru_3 (GRU)                  (None, 100)               60300     
_________________________________________________________________
dense_9 (Dense)              (None, 8)                 808       
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 9         
=================================================================

Figure 6. GRU Model
