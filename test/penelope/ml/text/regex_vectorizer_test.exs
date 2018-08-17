defmodule Penelope.ML.Text.RegexVectorizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.RegexVectorizer
  alias Penelope.ML.Vector

  test "transform" do
    x = ["", "sentence matches", "sentence does not match"]
    regexes = [~r/matches/, ~r/sentence/, ~r/^match/]

    expect = [[0, 0, 0], [1, 1, 0], [0, 1, 0]]
    x = RegexVectorizer.transform(%{regexes: regexes}, %{}, x)
    assert x === Enum.map(expect, &Vector.from_list/1)
  end
end
