# Content-Moderation-of-Hinglish-Comments

This repository contains multiple approaches for Hate Speech Detection in Hindi-English Code-Mixed Data. The task is to perform a multi-class classification of the Hinglish data into three classes : Positive, Negative and Neutral.

Method 1 : Profanity-weighted approach

This method takes care of Hinglish profane words in a sentence and thus generates powerful sentence embeddings specifically for Hinglish content.

Method 2 : Concatenated P Means approach

Fine-tuning a FastText model on a large Hinglish dataset can generate good word embeddings, and by concatenating different means, we can generate good sentence vectors.

Comparison with basic Deep Learning models :

Existing deep neural network architectures (CNN 1D, LSTM, BiLSTM, GRUs) are not robust enough for classification of hate speech data in code-mixed language.

Conclusion :

A fusion of models inclined for generating powerful sentence embeddings and the existing baseline deep learning models can provide significant improvement in the overall performance of Hinglish Hate Speech Detection Task.

![Poster presentation](https://user-images.githubusercontent.com/66362309/131038711-c30a57c5-e4f8-4c25-8ec6-1bd2476a5d9a.png)
