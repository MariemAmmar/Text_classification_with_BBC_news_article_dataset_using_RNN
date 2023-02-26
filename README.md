# BBC News Classification using Tensorflow
This code is a simple example of how to use Tensorflow to classify BBC news articles into 6 categories: business, entertainment, politics, sport, tech, and other. The dataset used in this code is the BBC Text Dataset.
This is a code implementation of text classification using an RNN model to classify BBC news articles. The code is written in Python using TensorFlow and several libraries including NLTK, Keras, and Matplotlib.
The implementation starts with importing the required libraries and loading the data, which is stored in a CSV file. Then, data cleaning, preprocessing, and preparation steps are performed, including removing stopwords, splitting the data into training and validation sets, tokenizing and padding the sequences, and tokenizing the labels.
After that, the RNN model is built and trained using the training and validation sets, and the training history is plotted to ensure there is no overfitting. Finally, a new text is tested using the trained model to predict its label.




## Going Deeper with the code 

This code reads in a CSV file containing news article text and their corresponding labels (categories), preprocesses the text data by removing stopwords, tokenizes the text data, and pads the tokenized sequences to a fixed length. It then tokenizes the labels and trains a neural network model to predict the label/category of a given news article. The model architecture consists of an embedding layer, a global average pooling layer, a dense layer with ReLU activation, and a final dense layer with a softmax activation.

The code begins by importing necessary packages including csv, tensorflow, numpy, nltk, stopwords from the nltk package, Tokenizer and pad_sequences from tensorflow.keras.preprocessing.text. The csv package is used to read in the CSV file, while numpy is used to convert the label sequences to numpy arrays. nltk and stopwords are used to preprocess the text data by removing stopwords, and Tokenizer and pad_sequences are used to tokenize the text data and pad the tokenized sequences to a fixed length.

The code then initializes several parameters including vocab_size, embedding_dim, max_length, trunc_type, padding_type, oov_tok, and training_portion. vocab_size determines the number of words to keep in the vocabulary, embedding_dim determines the size of the embedding vectors, max_length sets the maximum length of each sequence (which is achieved through padding), trunc_type and padding_type determine how to truncate or pad sequences that are longer or shorter than max_length, and oov_tok is a token used to represent out-of-vocabulary words. training_portion determines the proportion of the data that will be used for training, with the remainder used for validation.

The code then reads in the CSV file using the csv package and processes each row by checking that the row has at least two columns, appending the label to a list called labels, and processing the text by removing stopwords and appending the cleaned text to a list called sentences.

The code then splits the data into training and validation sets using the train_size variable, and tokenizes the training and validation sequences using Tokenizer from tensorflow.keras.preprocessing.text. It also creates a dictionary called word_index that maps words to their respective indices in the tokenizer, and pads the tokenized sequences using pad_sequences from tensorflow.keras.preprocessing.sequence.

The code then tokenizes the labels using another Tokenizer, and converts the label sequences to numpy arrays using numpy.

Finally, the code builds a neural network model using Sequential from tensorflow.keras, with an embedding layer, a global average pooling layer, a dense layer with ReLU activation, and a final dense layer with a softmax activation. It then compiles the model using sparse_categorical_crossentropy loss, adam optimizer, and accuracy metric. The model is trained for a specified number of epochs using fit and the training and validation data, and the training history is stored in a history variable.
