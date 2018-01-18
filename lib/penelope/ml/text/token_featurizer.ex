defmodule Penelope.ML.Text.TokenFeaturizer do
  @moduledoc """
  The token featurizer converts a list of tokenized documents into
  a map per token, in the format used for sequence classification.
  """

  @doc """
  transforms the sequence feature list
  """
  @spec transform(model :: map, context :: map, x :: [[String.t()]]) :: [
          [map]
        ]
  def transform(_model, _context, x) do
    Enum.map(x, &Enum.map(&1, fn x -> %{x => 1.0} end))
  end
end
