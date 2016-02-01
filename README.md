# Authorific: Machine learning for dictators and Taylor Swift
## Hack Cambridge Project
## Max Conway, Natasha Latysheva, Alexey Morgunov, Andrey Malinin

## Support vector machine classifier for categorizing text data

Text data is full of distinctive vocabulary and conceptual signals. These signals can be used to perform predictive tasks, like classifying raw text into categories. 

We should be able to, in theory, analyse a piece of text and make quantitatively-backed up statements such as "this looks like an Adele song", or "seems like something Hitler would write", or "sounds like that one Taylor Swift song with the mercenary ladies". 

If we are able to generate meaningful features from the text, then even unstructured, messy text data can be treated as a just another type of input into a machine learning algorithm. 

Here, we use a dual-pronged ***deep learning*** and ***support vector machine*** approach to extract concept features and vocabulary from raw text and use trained models to classify new text that a user inputs (specialised repo for web app: https://github.com/maxconway/authorific). 

We build up the models on the training set, and achieve very high accuracies on the test set (over 95% for the deep neural nets and over 90% accuracy in the case of SVMs). 

## How do deep neural networks and SVMs allows us to predict if your supervisor's emails resemble Hitler's speeches?

Neural networks (NNs) operate by stacking multiple artificial neurons, which are simple processing units, into a layer and then stacking layer upon layer. In this way, the network can learn to model very complex representations of the training data.

Support vector machines (SVMs) are another type of machine learning algorithm that turn out to be very good at classifying text. 

By training our NN and SVM models  on a set of text where we know the labels, we can use the optimised models to predict the labels of the test data set, which allows us to estimate the accuracy and usefulness of our models.

More interestingly, we can use these trained models to classify new text which is input by a user into a web application.

Although we will not go into massive detail here, in SVMs in SVMs the following equation gives a value proportional to the distance from the margin. We wish to minimize it:

## The intuition behind word frequency matrices and SVMs

Both the neural networks and support vector machines are fitted to matrices of features, and a common type of feature are word frequency matrices. In practice, these matrices are normalised in various ways (see td-df matrices).

Intuitively, words that occur frequently and relatively uniquely in one category of text, such as the words “Heil” and or the 2 word pair (2-gram) "German peoples” are likely to belong to the "Hitler" category of speeches and should be predictive of that category. 

One common way to formalize this notion is by using a ‘bag of words’ approach that tokenizes, counts and scales words frequencies across categories, which can then be used as the starting point for training predictive algorithms for classifying new instances of text. xt.

Naïve Bayes (NB) and support vector machines (SVM) are classic machine learning methods which have been successfully applied to the large-scale and sparse problems often encountered in text classification and natural language processing. 

We applied a feedforward neural network, Gaussian Naive Bayes and a linear SVM to a bag of words representation of various categories of text in order to attempt to predict categories from user-supplied input text.

