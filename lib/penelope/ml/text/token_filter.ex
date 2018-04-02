defmodule Penelope.ML.Text.TokenFilter do
  @moduledoc """
  The the token filter removes stopwords from a token list in the pipeline.
  """

  @doc """
  removes stop word tokens from the document stream
  """
  @spec transform(
          model :: %{stopwords: [String.t()]},
          context :: map,
          x :: [[String.t()]]
        ) :: [[String.t()]]
  def transform(model, _context, x) do
    Enum.map(x, &filter_tokens(&1, model.stopwords))
  end

  defp filter_tokens(x, stopwords) do
    Enum.reject(x, &(&1 in stopwords))
  end
end
