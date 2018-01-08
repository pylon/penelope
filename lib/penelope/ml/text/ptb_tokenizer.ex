defmodule Penelope.ML.Text.PTBTokenizer do
  @moduledoc """
  This pipeline component adapts the treebank tokenizer to the pipeline
  transformer conventions. It produces a sequence of tokens for each
  incoming document string.
  """

  alias Penelope.NLP.PennTreebankTokenizer

  @doc """
  transforms a list of documents into a list of lists of tokens
  """
  @spec transform(model::map, context::map, x::[String.t]) :: [[String.t]]
  def transform(_model, _context, x) do
    Enum.map(x, &PennTreebankTokenizer.tokenize/1)
  end
end
