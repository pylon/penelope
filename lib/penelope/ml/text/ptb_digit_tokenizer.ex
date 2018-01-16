defmodule Penelope.ML.Text.PTBDigitTokenizer do
  @moduledoc """
  This pipeline component adapts the treebank tokenizer + the digit token
  preprocessor to the pipeline transformer conventions. It produces a
  sequence of tokens for each incoming document string.
  """

  alias Penelope.NLP.Tokenize.PennTreebankTokenizer

  @doc """
  transforms a list of documents into a list of token lists
  transforms a list of token lists into a list of documents
  """
  @spec transform(model::map, context::map, x::[String.t]) :: [[String.t]]
  def transform(_model, _context, x) do
    Enum.map(x, &do_transform/1)
  end

  defp do_transform(x) when is_binary(x) do
    x
    |> String.replace(~r/([\d])/, "\\1 ")
    |> String.replace(~r/([\.])([\d])/, "\\1 \\2")
    |> PennTreebankTokenizer.tokenize()
  end
  defp do_transform(x) when is_list(x) do
    x
    |> PennTreebankTokenizer.detokenize()
    |> String.replace(~r/([\d])\s(?=[\d])/, "\\1")
    |> String.replace(~r/([\d])\s([\.])/, "\\1\\2")
  end
end
