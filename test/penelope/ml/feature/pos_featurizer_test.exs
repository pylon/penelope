defmodule Penelope.ML.Text.POSFeaturizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.POSFeaturizer

  test "transform" do
    x = []
    expect = []

    assert POSFeaturizer.transform(%{}, %{}, x) === expect

    x = [[], ["It", "is", "a", "little-known", "fact1"]]

    expect = [
      [],
      [
        %{
          "has_hyphen" => false,
          "has_digit" => false,
          "has_cap" => true,
          "pre_1" => "I",
          "pre_2" => "It",
          "pre_3" => "It",
          "pre_4" => "It",
          "suff_1" => "t",
          "suff_2" => "It",
          "suff_3" => "It",
          "suff_4" => "It",
          "tok_-2" => "",
          "tok_-1" => "",
          "tok_0" => "It",
          "tok_1" => "is",
          "tok_2" => "a"
        },
        %{
          "has_hyphen" => false,
          "has_digit" => false,
          "has_cap" => false,
          "pre_1" => "i",
          "pre_2" => "is",
          "pre_3" => "is",
          "pre_4" => "is",
          "suff_1" => "s",
          "suff_2" => "is",
          "suff_3" => "is",
          "suff_4" => "is",
          "tok_-2" => "",
          "tok_-1" => "It",
          "tok_0" => "is",
          "tok_1" => "a",
          "tok_2" => "little-known"
        },
        %{
          "has_hyphen" => false,
          "has_digit" => false,
          "has_cap" => false,
          "pre_1" => "a",
          "pre_2" => "a",
          "pre_3" => "a",
          "pre_4" => "a",
          "suff_1" => "a",
          "suff_2" => "a",
          "suff_3" => "a",
          "suff_4" => "a",
          "tok_-2" => "It",
          "tok_-1" => "is",
          "tok_0" => "a",
          "tok_1" => "little-known",
          "tok_2" => "fact1"
        },
        %{
          "has_hyphen" => true,
          "has_digit" => false,
          "has_cap" => false,
          "pre_1" => "l",
          "pre_2" => "li",
          "pre_3" => "lit",
          "pre_4" => "litt",
          "suff_1" => "n",
          "suff_2" => "wn",
          "suff_3" => "own",
          "suff_4" => "nown",
          "tok_-2" => "is",
          "tok_-1" => "a",
          "tok_0" => "little-known",
          "tok_1" => "fact1",
          "tok_2" => ""
        },
        %{
          "has_hyphen" => false,
          "has_digit" => true,
          "has_cap" => false,
          "pre_1" => "f",
          "pre_2" => "fa",
          "pre_3" => "fac",
          "pre_4" => "fact",
          "suff_1" => "1",
          "suff_2" => "t1",
          "suff_3" => "ct1",
          "suff_4" => "act1",
          "tok_-2" => "a",
          "tok_-1" => "little-known",
          "tok_0" => "fact1",
          "tok_1" => "",
          "tok_2" => ""
        }
      ]
    ]

    assert POSFeaturizer.transform(%{}, %{}, x) === expect
  end
end
