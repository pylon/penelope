# Penelope

Natural Language Processing (NLP) and Machine Learning (ML) library for Elixir.
Penelope provides a scikit-learn-inspired interface to the the
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/),
[LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/), and
[CRFsuite](http://www.chokkan.org/software/crfsuite/) C/C++ libraries in
Elixir, which can be used for many ML/NLP applications.

## Status
[![Hex](http://img.shields.io/hexpm/v/penelope.svg?style=flat)](https://hex.pm/packages/penelope)
[![CircleCI](https://circleci.com/gh/pylon/penelope.svg?style=shield)](https://circleci.com/gh/pylon/penelope)
[![Coverage](https://coveralls.io/repos/github/pylon/penelope/badge.svg)](https://coveralls.io/github/pylon/penelope)

The API reference is available [here](https://hexdocs.pm/penelope/).

## Installation

### Dependencies
First, clone the project's submodules.

```bash
git submodule update --init
```

This package requires an implementation of `BLAS` for efficient matrix math.
It can be installed on each platform as follows:

#### OSX
BLAS is built into OSX.

#### Alpine
Install `openblas-dev` via apk.

```bash
sudo apk add openblas-dev
```

#### Ubuntu
Install `libblas-dev` via apt.

```bash
sudo apt install libblas-dev
```

### Hex
```elixir
def deps do
  [
    {:penelope, "~> 0.4"}
  ]
end
```

## Usage

### Intent Classification/Entity Recognition
Penelope can be used to build a machine learning model for identifying natural
language utterances and extracting parameters from them. The
`Penelope.NLP.IntentClassifier` module uses a predictor pipeline for
recognizing intents and a recognizer pipeline for extracting named entities
from the utterance. The following is a contrived example that classifies
intents based on the token length of the utterance.

```elixir
alias Penelope.NLP.IntentClassifier

pipeline = %{
  tokenizer: [{:ptb_tokenizer, []}],
  classifier: [{:count_vectorizer, []},
               {:linear_classifier, [probability?: true]}],
  recognizer: [{:crf_tagger, []}],
}
x = [
  "you have four pears",
  "three hundred apples would be a lot"
]
y = [
  {"intent_1", ["o", "o", "b_count", "b_fruit"]},
  {"intent_2", ["b_count", "i_count", "b_fruit", "o", "o", "o", "o"]}
]
classifier = IntentClassifier.fit(%{}, x, y, pipeline)

{intents, params} = IntentClassifier.predict_intent(
  classifier,
  %{},
  "I have three bananas"
)
```

#### Pipeline Definition

```elixir
pipeline = %{
  tokenizer: [{:ptb_tokenizer, []}],
  classifier: [{:count_vectorizer, []},
               {:linear_classifier, [probability?: true]}],
  recognizer: [{:crf_tagger, []}],
}
```

This block configures the tokenizer, classifier, and recognizer pipelines
used by the intent classifier. A pipeline in Penelope is a list of components
and configuration that are used to train/predict a machine learning model,
with an interface similar to that used in scikit-learn.

The tokenizer converts a string utterance into a sequence of tokens. In this
example, we use the [Penn Treebank](ftp://ftp.cis.upenn.edu/pub/treebank/public_html/tokenization.html)
tokenizer (`:ptb-tokenizer`). The tokenizer pipeline is run before either
of the other two pipelines, so that they can share its output.

The classifier pipeline receives a tokenized utterance (**x**) and class
labels (**y**) and learns a model that can predict the label from the
utterance. In this example, we use a simple token count vectorizer (number
of tokens in the utterance) and a [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
classifier to predict the class labels.

Finally, the recognizer pipeline receives a tokenized sequence (**x**) and
sequence tags (**y**) to learn a model that can predict the label of each
tag in the sequence. This allows the recognizer to extract `slot` values from
natural language utterances. This example uses a Conditional Random Field
([CRF](https://en.wikipedia.org/wiki/Conditional_random_field)) model, which
can be thought of as a sequence extension of logistic regression, to tag
the tokens in the utterance.

#### Training

```elixir
x = [
  "you have four pears",
  "three hundred apples would be a lot"
]
y = [
  {"intent_1", ["o", "o", "b_count", "b_fruit"]},
  {"intent_2", ["b_count", "i_count", "b_fruit", "o", "o", "o", "o"]}
]
classifier = IntentClassifier.fit(%{}, x, y, pipeline)
```

Inputs (**x**) to the intent classifier are simple natural language
utterances. These inputs are tokenized and converted to feature vectors/maps
as needed by the classifier/recognizer.

Each label (**y**) is a tuple of `{intent, tags}`, where `intent` is the
class label of the intent for the corresponding **x** value. `tags` is a
list of token tags, each of which is a label for the corresponding token
in the utterance **x**. Tag labels are expressed using the
Inside-Outside-Beginning ([IOB](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_%28tagging%29))
format. In the above snippet, the following are the token tags for the first
utterance.

token|tag
-|-
you|o
have|o
four|b_count
pears|b_fruit

#### Prediction

```elixir
{intents, params} = IntentClassifier.predict_intent(
  classifier,
  %{},
  "I have three bananas"
)
```

The snippet above returns the following `intents` map and `params` map that
classify the utterance. The `intents` map contains the posterior probability
of each intent, all of which sum to 1.0. The `params` map contains the
map of entity names extracted from the utterance, based on the names
specified in the training examples.

```elixir
{
    %{
        "intent_1" => 0.6666666661872298,
        "intent_2" => 0.3333333338127702
    },
    %{
        "count" => "three",
        "fruit" => "bananas"
    }
}
```

#### Improvements

Obviously, using the token count as the only feature to try to predict
an intent is silly, and using only the input tokens to train the entity
recognizer will not generalize well. For better classification/recognition,
Penelope includes several feature generation components/vectorizers, including
support for pretrained embeddings (word vectors) and regexes. Examples of
these can be found in the API reference.

## License

Copyright 2017 Pylon, Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
