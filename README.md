# Natural Language Processing sentence pair comparison and classification 
This is part of a larger project that cannot be shared due to propreitary reasons. Here, a SVM baseline is established for the classification of a sentence-pair dataset to check for similarities and contradictions in documents. The code for the Streamlit based webapp and the pipeline for training and validation, along with the SVM model is added to this repository.

## Bert based models

The Albert and Legal-Bert cannot be shared as it is was developed and paid for by the Company. For this, Pytorch and Hugging Face transformer library has been used for the implementation. The
code was modified to use with the Google Colab Pro GPUs to save time. The cleaned dataset was mounted from the author’s google drive and access needs to be granted for it to be used. Each trained epoch is saved as a checkpoint to avoid loss of data and the code was modified to enable this and to load the chosen epoch for testing as well. 

## The SVM Model Implmentation (Linear SVC) - Baseline

After initially experimenting with various options on a smaller subset of the training dataset (10K)
and RandomisedSearchCV, The Linear SVC was chosen as the SVM model to tune. LinearSVC was preferred over SVC with the linear kernel as it scales better to large numbers of samples
(Scikitlearn 2021).

![image](https://github.com/Surya-LR/NaturalLanguageProcessing_SVM_Baseline/assets/77691667/a39f8ddf-4d31-4428-ada5-39a9fc0b58f7)

The tuning of the LinearSVC hyperparameter ’C’ was done can be seen that a ‘C’ value of 1 gives the best results on the validation set and is
used in the final implementation of the LinearSVC code.

## Dataset
Data used was The Stanford Natural Language Inference (SNLI) Corpus from https://nlp.stanford.edu/projects/snli/.The corpus is a collection of
over 570,000 sentence pairs ( a premise-hypothesis pair) written by humans. The relationship
between the sentences in the sentence pairs has been manually labeled by four annotators.Only the labels with consensus were used
in this project.

## Text pre-processing

The raw text from the SNLI dataset was loaded into a dataframe. The columns(Premise and Hypothesis) are passed to a function for cleaning the dataframe, renaming the attributes, and
dropping unwanted columns. This function is shown in Figure below:

![image](https://github.com/Surya-LR/NaturalLanguageProcessing_SVM_Baseline/assets/77691667/55db6317-f3db-4399-ad7e-d7654a0a4c2f)


It is then passed to a pre-processing function. This function cleans the string, converts it to lower-case, and applies
lemmatization.

![image](https://github.com/Surya-LR/NaturalLanguageProcessing_SVM_Baseline/assets/77691667/86380f6c-11df-46ef-90fe-615056920e19)


As the algorithms cannot work with the raw text directly, the text must be converted into vectors of numbers using a text vectorizer. TF-IDF(Term frequency-inverse document frequency),
is used in this project to transform the raw text into a usable vector.TF-IDF is a combination of Term Frequency (TF) and Inverse Document Frequency (IDF). These concepts together indicate
how common and frequent the word in the document is. When passed through this vectorizer the text is converted to a matrix of TF-IDF features. The features from the sentences are combined with the metadata in the feature union section
with the help of a pipeline and sent to the SVM model. 






