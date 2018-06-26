defmodule Penelope.NLP.POSTagger do
  @moduledoc """
  The part-of-speech tagger transforms a tokenized sentence into a list of
  `{token, pos_tag}` tuples. The tagger takes no responsibility for
  tokenization; this means that callers must be careful to maintain the same
  tokenization scheme between training and evaluating to ensure the best
  results.

  As this tagger does not ship with a pretrained model, it is both
  language- and tagset-agnostic, though the default feature set used
  (see `POSFeaturizer`) was designed for English.

  See `POSTaggerTrainer.train/2` for an example
  of how to train a new POS tagger model.
  """

  alias Penelope.ML.Pipeline

  @type model :: %{
          tagger: [{atom, any}],
          featurizer: [{atom, any}]
        }

  @doc """
  Fits the tagger model. The following keys may be fed to `pipelines`
  to include custom components in the tagger pipeline:

  |key               |default                       |
  |------------------|------------------------------|
  |`featurizer`      |`[{:pos_featurizer, []}]`     |
  |`tagger`          |`[{:crf_tagger, []}]`         |
  """
  @spec fit(
          context :: map,
          x :: [tokens :: [String.t()]],
          y :: [tags :: [String.t()]],
          pipelines :: keyword
        ) :: model
  def fit(context, x, y, pipelines \\ []) do
    featurizer_config =
      Keyword.get(pipelines, :featurizer, [{:pos_featurizer, []}])

    tagger_config = Keyword.get(pipelines, :tagger, [{:crf_tagger, []}])

    featurizer = Pipeline.fit(context, x, y, featurizer_config)
    features = Pipeline.transform(featurizer, context, x)

    %{
      tagger: Pipeline.fit(context, features, y, tagger_config),
      featurizer: featurizer
    }
  end

  @doc """
  Attaches part of speech tags to a list of tokens.

  Example:
  ```
  iex> POSTagger.tag(model, %{}, ["Judy", "saw", "her"])
  [{"Judy", "NNP"}, {"saw", "VBD"}, {"her", "PRP$"}]
  ```
  """
  @spec tag(model :: model, context :: map, tokens :: [String.t()]) :: [
          {String.t(), String.t()}
        ]
  def tag(model, context, tokens) do
    features = Pipeline.transform(model.featurizer, context, [tokens])

    [{tags, _probability}] =
      Pipeline.predict_sequence(model.tagger, context, features)

    Enum.zip(tokens, tags)
  end

  @doc """
  Imports parameters from a serialized model.
  """
  @spec compile(params :: map) :: model
  def compile(params) do
    %{
      tagger: Pipeline.compile(params["tagger"]),
      featurizer: Pipeline.compile(params["featurizer"])
    }
  end

  @doc """
  Exports a runtime model to a serializable data structure.
  """
  @spec export(model :: model) :: map
  def export(model) do
    %{
      "tagger" => Pipeline.export(model.tagger),
      "featurizer" => Pipeline.export(model.featurizer)
    }
  end
end
