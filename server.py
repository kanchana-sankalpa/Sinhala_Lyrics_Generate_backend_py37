from textgenrnn import textgenrnn
import numpy as np
import tensorflow as tf
import sampler
#import regex


def get_model_api():
    '''Return lambda function for API'''
    # 1 Initilize model once and for all and reload weights

    #config_path = 'weights/Sew_file1_V1/model_V2_config.json'
    config_path = 'weights/Sew_file1_V1/sew_v2_config.json'
    vocab_path = 'weights/Sew_file1_V1/sew_v2_vocab.json'
    weights_path = 'weights/Sew_file1_V1/sew_v2_weights.hdf5'

    textgen = textgenrnn(config_path=config_path,
                         vocab_path=vocab_path,
                         weights_path=weights_path)
    textgen.generate() #resolved a memory addressing bug of keras, DO NOT remove

    def model_api(input_data):
        # 2. pre-process input
        punc = ["(", ")", "[",
                "]"]  # FIXME: add other cleaning if necessary, check if not redundant with library cleaning
        prefix = "".join(c.lower() for c in input_data if c not in punc)

        # 3.0 initialize generation parameters
        temperatures = [0.5, 0.6, 0.7] #TODO: to tweak.
        num_line = 5
        prefix_mode = 2  # see doc of sampler.py for mode 0,1,2
        prefix_proba = 0.5

        # 3.1 call model predict function
        prediction = sampler.lyrics_generator(textgen, prefix,
                                              temperatures=temperatures, num_line=num_line,
                                              prefix_mode=prefix_mode, prefix_proba=prefix_proba)

        # 4. process the output
        output_data = {"output": prediction}

        # 5. return the output for the api
        print(output_data)
        return output_data

    return model_api



#reg_output= regex.findall(r'\X', corpus)


def get_tpu_model_api():
    f=open("datasets\sew_samadhi_improved.txt", "r" , encoding="utf-8")
    reg_output = f.read()
    print('output from the text file read')
    #print(reg_output)
    duplicates_removed = list(dict.fromkeys(reg_output))
    #print (duplicates_removed)

    chars = duplicates_removed
    #print (chars)
    #chars.pop(3)
    #print (chars)
    char_indices = dict((c, i) for i, c in enumerate(duplicates_removed))
    print(char_indices)
    indices_char = dict((i, c) for i, c in enumerate(duplicates_removed))
    print(indices_char)

    '''Return lambda function for API'''
    # 1 Initilize model once and for all and reload weights

    BATCH_SIZE = 5
    PREDICT_LEN = 250

# Keras requires the batch size be specified ahead of time for stateful models.
# We use a sequence length of 1, as we will be feeding in one character at a 
# time and predicting the next character.
    prediction_model = lstm_model(seq_len=1, batch_size=BATCH_SIZE, stateful=True)
    prediction_model.load_weights('weights/TPU1/bard.h5')

    # We seed the model with our initial string, copied BATCH_SIZE times
    def model_tpu_api(input_data):
        #seed_txt = 'පෙරදා වින්දේ මා පමණය'
        seed_txt = input_data
        seed = transform(seed_txt,char_indices)
        seed = np.repeat(np.expand_dims(seed, 0), BATCH_SIZE, axis=0)

        # First, run the seed forward to prime the state of the model.
        prediction_model.reset_states()
        for i in range(len(seed_txt) - 1):
            #print(seed[:, i:i + 1])
            prediction_model.predict(seed[:, i:i + 1])

        # Now we can accumulate predictions!
        predictions = [seed[:, -1:]]
        for i in range(PREDICT_LEN):
            last_word = predictions[-1]
            next_probits = prediction_model.predict(last_word)[:, 0, :]
  
            # sample from our output distribution
            next_idx = [
                np.random.choice(256, p=next_probits[i])
                for i in range(BATCH_SIZE) 
            ]
            predictions.append(np.asarray(next_idx, dtype=np.int32))
  
        all_results='';
        for i in range(BATCH_SIZE):
            print('PREDICTION %d\n\n' % i)
            p = [predictions[j][i] for j in range(PREDICT_LEN)]
            print(p)
            temp = [indices_char.get(c) for c in p[1:]]
            # add p[1:] to skip frist element in array 
            #generated = ''.join([chr(c) for c in p])  # Convert back to text
            generated = ''.join([indices_char.get(c) for c in p[1:]])
            print(generated)
            # 5. return the output for the api
            #return generated
            all_results=all_results+generated

        output_data = {"output": all_results}
        return output_data
    return model_tpu_api


EMBEDDING_DIM = 512

def lstm_model(seq_len=100, batch_size=None, stateful=True):
  """Language model: predict the next word given the current word."""
  source = tf.keras.Input(
      name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

  embedding = tf.keras.layers.Embedding(input_dim=256, output_dim=EMBEDDING_DIM)(source)
  lstm_1 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(embedding)
  lstm_2 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_1)
  predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='softmax'))(lstm_2)
  return tf.keras.Model(inputs=[source], outputs=[predicted_char])



def transform(txt,char_indices):
  #for element in txt:
    #print(element)
    #print(ord(element))
    #if ord(c) < 255
  return np.asarray([int(char_indices.get(c)) for c in txt ], dtype=np.int32)
  #return np.asarray([ord(c) for c in txt ], dtype=np.int32)