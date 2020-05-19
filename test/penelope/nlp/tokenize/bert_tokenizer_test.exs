defmodule Penelope.NLP.Tokenize.BertTokenizerTest do
  @moduledoc false

  use ExUnit.Case
  alias Penelope.NLP.Tokenize.BertTokenizer, as: Tokenizer

  @vocab [
           "[PAD]",
           "[UNK]",
           "[CLS]",
           "[SEP]",
           "[MASK]",
           "-",
           "'",
           "\"",
           "$",
           "b",
           "e",
           "ē",
           "r",
           "t",
           "be",
           "##rt",
           "ber",
           "##t"
         ]
         |> Enum.with_index()
         |> Map.new()

  test "examples" do
    examples = %{
      # empty
      "" => {[], [], []},

      # tokenize
      "be  \t\r\nrt" => {["be", "rt"], [0, 1, 1], ["be", "r", "##t"]},
      "be\u200Art" => {["be", "rt"], [0, 1, 1], ["be", "r", "##t"]},
      " bert " => {["bert"], [0, 0], ["ber", "##t"]},

      # normalize
      "BERT" => {["BERT"], [0, 0], ["ber", "##t"]},
      "bért" => {["bért"], [0, 0], ["ber", "##t"]},

      # strip
      "be\u0301\u0000\uFFFDrt" =>
        {["be\u0301\u0000\uFFFDrt"], [0, 0], ["ber", "##t"]},

      # split
      "[PAD]" => {["[PAD]"], [0], ["[PAD]"]},
      "be-rt" => {["be-rt"], [0, 0, 0, 0], ["be", "-", "r", "##t"]},
      "be'rt" => {["be'rt"], [0, 0, 0, 0], ["be", "'", "r", "##t"]},
      "be$rt" => {["be$rt"], [0, 0, 0, 0], ["be", "$", "r", "##t"]},

      # piece
      "berrtt" => {["berrtt"], [0, 0, 0], ["ber", "##rt", "##t"]},

      # fail
      "be🔥rt" => {["be🔥rt"], [0], ["[UNK]"]}
    }

    for {input, {tokens, indexes, keys}} <- examples do
      expect = {tokens, indexes, Enum.map(keys, &@vocab[&1])}
      encoded = Tokenizer.encode(input, @vocab)
      assert encoded === expect

      # the tokenizer should be lossless for everything except spaces
      expect = input |> String.replace(~r/\s+/u, " ") |> String.trim()
      assert Tokenizer.decode(encoded) === expect
    end
  end
end
