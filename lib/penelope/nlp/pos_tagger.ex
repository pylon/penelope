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

  @type model :: %{pos_tagger: [{atom, any}]}

  @doc """
  Fits the tagger model. Custom featurizers may be supplied.
  """
  @spec fit(
          context :: map,
          x :: [tokens :: [String.t()]],
          y :: [tags :: [String.t()]],
          featurizers :: [{atom | String.t(), [any]}]
        ) :: model
  def fit(context, x, y, featurizers \\ [{:pos_featurizer, []}]) do
    pipeline = featurizers ++ [{:crf_tagger, []}]
    %{pos_tagger: Pipeline.fit(context, x, y, pipeline)}
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
    [{tags, _probability}] =
      Pipeline.predict_sequence(model.pos_tagger, context, [tokens])

    Enum.zip(tokens, tags)
  end

  @doc """
  Imports parameters from a serialized model.
  """
  @spec compile(params :: map) :: model
  def compile(params),
    do: %{pos_tagger: Pipeline.compile(params["pos_tagger"])}

  @doc """
  Exports a runtime model to a serializable data structure.
  """
  @spec export(model :: model) :: map
  def export(model),
    do: %{"pos_tagger" => Pipeline.export(model.pos_tagger)}
end
