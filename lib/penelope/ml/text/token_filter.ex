defmodule Penelope.ML.Text.TokenFilter do
  @moduledoc """
  The the token filter removes tokens from a token list in the pipeline.
  """

  @doc """
  removes stop word tokens from the document stream
  """
  @spec transform(
          model :: %{tokens: [String.t()]},
          context :: map,
          x :: [[String.t()]]
        ) :: [[String.t()]]
  def transform(model, _context, x) do
    Enum.map(x, &filter_tokens(&1, model.tokens))
  end

  defp filter_tokens(x, tokens) do
    Enum.reject(x, &(&1 in tokens))
  end
end
