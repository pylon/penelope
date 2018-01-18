defmodule Penelope.ML.Text.CountVectorizer do
  @moduledoc """
  The CountVectorizer simply counts the number of tokens in the incoming
  documents. It assumes that samples have already been tokenized into
  a list per sample. This vectorizer is useful for biasing a model for
  longer/shorter documents.
  """

  alias Penelope.ML.Vector

  @doc """
  transforms a list of samples (list of lists of tokens) into vectors
  """
  @spec transform(model :: map, context :: map, x :: [[String.t()]]) :: [
          Vector.t()
        ]
  def transform(_model, _context, x) do
    Enum.map(x, fn x -> Vector.from_list([Enum.count(x) / 1]) end)
  end
end
