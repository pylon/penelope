defmodule Penelope.NLP.Tokenize.Tokenizer do
  @moduledoc """
    The behaviour implemented by all tokenizers.
  """

  @doc """
  Separate a string into a list of tokens.
  """
  @callback tokenize(String.t) :: [String.t]

  @doc """
  Reverse the tokenization process, turning a list of
  tokens into a single string. Tokenization is often
  a lossy process, so detokenization is not guaranteed
  to return a string identical to the original tokenizer's
  input.
  """
  @callback detokenize([String.t]) :: String.t

end
