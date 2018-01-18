defmodule Penelope.ML.Text.LowercasePreprocessor do
  @moduledoc """
  downcasing document preprocessor
  """

  @doc """
  transforms a list of documents into lowercase documents
  """
  @spec transform(model :: map, context :: map, x :: [String.t()]) :: [
          String.t()
        ]
  def transform(_model, _context, x) do
    Enum.map(x, &String.downcase/1)
  end
end
