from tensorflow import keras
from filetochars import char_indices, file_to_text, indices_char, text_to_chars
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import numpy as np
import random
import io




def lstm_character(lang):
    text=file_to_text(lang+'-train.txt')
    ttext=file_to_text(lang+'-test.txt')
    chars=text_to_chars(text)
    tchars=text_to_chars(ttext)
    charindices=char_indices(chars)
    tcharindices=char_indices(tchars)
    indiceschar=indices_char(chars)
    tindiceschar=indices_char(tchars)


# cut the text in semi-redundant sequences of maxlen characters
    maxlen = 10
    step = 2
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i + maxlen])
    print("Number of sequences:", len(sentences))

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, charindices[char]] = 1
        y[i, charindices[next_chars[i]]] = 1

   
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state = 42)

    """
## Build the model: a single LSTM layer
"""

    model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(128),
        layers.Dense(len(chars), activation="softmax"),
    ]
)
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['accuracy'])

    """
## Prepare the text sampling function
"""


    def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    """
## Train the model
"""

    epochs = 10
    batch_size = 128

    for epoch in range(epochs):
        model.fit(x_train, y_train, epochs=2, validation_data=(x_dev, y_dev),verbose=2)
        print()
        print("Generating text after epoch: %d" % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            generated = ""
            sentence = text[start_index : start_index + maxlen]
            print('...Generating with seed: "' + sentence + '"')

            for i in range(100):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, charindices[char]] = 1.0
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indiceschar[next_index]
                sentence = sentence[1:] + next_char
                generated += next_char

            print("...Generated: ", generated)
            print()
            