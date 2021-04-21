# SinhalaLyrics-Back py 37 


## Getting started  Doc
### Setup

Setup a virtualenv and install the required libraries.

**Note**: Make sure you have a `python --version` == `python3.6` otherwise some library (especially `tensorflow` may not 
be available or behave as expected)

    
###  Train the model
The training is based on [minimaxir's textgenrnn](https://github.com/minimaxir/textgenrnn) with small pre-processing tweaks.
The development of this library is still active, you may want to use it instead of our fork:
    
    $ rm -r textgenrnn  
    $ pip install textgenrnn    
    
[train.py](train.py) contains the parameters and the path of the training dataset.  
Have a look at it and launch your own training.

**Parameters available**:   
*new_model*: True to create a new model or False to train based on an existing one.  
*rnn_layers*: Number of recurrent LSTM layers in the model (default: 2)  
*rnn_size*: Number of cells in each LSTM layer (default: 128)  
*rnn_bidirectional*: Whether to use Bidirectional LSTMs, which account for sequences both forwards and backwards. Recommended if the input text follows a specific schema. (default: False)  
*max_length*: Maximum number of previous characters/words to use before predicting the next token. This value should be reduced for word-level models (default: 40)  
*max_words*: Maximum number of words (by frequency) to consider for training (default: 10000)  
*dim_embeddings*: Dimensionality of the character/word embeddings (default: 100)  
*num_epochs*: number of epochs. (model is save at eah epoch by default)  
*word_level*: Whether to train the model at the word level (default: False)  
*dropout*: ratio of character to ignore on each sentence, may lead to better results. Don't use with word model  
*train_size*: ratio to use as training set. The remaining ratio will make a validation set. Default is 1. (no validation set)  
*gen_epochs*: during the training, a generation test will be made at each multiple of gen_epochs. Default is 1 (generation at each epoch)  
*max_gen_length*: length of the generation during training  

3 files will be generated by the training:  
*Model_name_config.json* : parameters given for the training  
*Model_name_vocab.json*  : vocabulary of the dataset  
*Model_name_weights.hdf5* : model weights   
Put these file in the weights folder.  

### Serve the model
The paths of the model files are defined in the first lines of [server.py](server.py).  
Change them to use your custom model.

The parameters of the custom generative function can also be tweaked.  
(documentation fot this function will be issued in another document)

To test the serving, launch the development flask server:
 
    $ python app.py
    
Make a request to the server in a different terminal:
  
    $ curl '127.0.0.1:5000/apiUS' -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "input=ඔබට මම"  
  
    {"output":"ඔබට මම ආදරය පෑ දවසේ\n ඔබ උන්නු තැන හදේ තාමත් හරිම උණුසුමයි"}

