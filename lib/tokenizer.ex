defmodule NlPotion.Tokenizer do
  @moduledoc """
    The behaviour implemented by all tokenizers.
  """

  @callback tokenize(String.t) :: [String.t]

end
