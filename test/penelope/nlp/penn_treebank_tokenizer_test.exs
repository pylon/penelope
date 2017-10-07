defmodule Penelope.NLP.PennTreebankTokenizerTest do
  @moduledoc false

  use ExUnit.Case
  alias Penelope.NLP.PennTreebankTokenizer, as: Tokenizer

  test "some sample tokenizations" do
    text = ~s(The quick, brown dog jumped over the "fence")
    expected = [
      "The",
      "quick",
      ",",
      "brown",
      "dog",
      "jumped",
      "over",
      "the",
      "``",
      "fence",
      "''"
    ]
    assert Tokenizer.tokenize(text) == expected
    text = "That costs $1.35."
    expected = ["That", "costs", "$", "1.35", "."]
    assert Tokenizer.tokenize(text) == expected
    text = "What're you gonna do about it?"
    expected = ["What", "'re", "you", "gon", "na", "do", "about", "it", "?"]
    assert Tokenizer.tokenize(text) == expected
    text = "I don't know!"
    expected = ["I", "do", "n't", "know", "!"]
    assert Tokenizer.tokenize(text) == expected
  end

end
