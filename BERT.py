import pandas as pd
from time import time
import tensorflow as tf
from collections import Counter
from sklearn.metrics import accuracy_score
# !pip install transformers  # importing pre-trained BERT model for NLP tasks
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures


# support functions
# =====================================================================================================================

def majority_vote(labels):
    labels = Counter(labels)
    if (labels['1'] > labels['0']) and (labels['1'] > labels['-1']):
        return 1
    elif (labels['0'] > labels['1']) and (labels['0'] > labels['-1']):
        return 0
    else:
        return -1


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


# main
# =====================================================================================================================

# path
kaggle = False  # change to True if training in Kaggle kernel
path = '../input/dit867-a3/' if kaggle else ''

# loading test set
df_test = pd.read_csv(path + 'a3_test_final.tsv', sep='\t', names=['label', 'doc'])

# loading training set
df_train = pd.read_csv(path + 'a3_train_final.tsv', sep='\t', names=['label', 'doc'])
N = len(df_train)

# transforming annotations to list
df_train['list'] = df_train['label'].str.split('/')

# applying majority vote
df_train['target'] = df_train['list'].apply(majority_vote)

# keeping only the rows for which there is a consensus on stance (unambiguous by majority)
df_train = df_train[df_train['target'] != -1][['target', 'doc']]
n = len(df_train)
print(f'Usable examples remaining in data set: {n:,}')
print(f'Percentage of data discarded due to ambiguity in label: {(1 - n / N) * 100:.2f}%')

# BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.summary()

# shortening train and val sets (delete after testing)
df_train = df_train.iloc[:100]
df_test = df_test.iloc[:100]

# tokenizing - transforming documents into input features accepted by BERT
train_InputExamples = df_train.apply(
    lambda x: InputExample(guid=None,
                           text_a=x['doc'],
                           text_b=None,
                           label=x['target']), axis=1)
train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

# model compiling and fitting
n_epochs = 2  # number of epochs to train
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
start = time()
model.fit(train_data, epochs=n_epochs)
print(f'Training complete after {round(time() - start):,} seconds.')

# predictions
x_test = tokenizer(df_test['doc'].tolist(), max_length=128, padding=True, truncation=True, return_tensors='tf')
output = model(x_test)  # model output
y_prob = tf.nn.softmax(output[0], axis=-1)  # label probabilities
y_pred = tf.argmax(y_prob, axis=1).numpy()  # model predictions

# test set accuracy
y_test = df_test['label'].to_numpy()
acc_test = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {acc_test * 100:.2f}%')
