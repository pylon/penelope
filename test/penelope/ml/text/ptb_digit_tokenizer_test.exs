defmodule Penelope.ML.Text.PTBDigitTokenizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.PTBDigitTokenizer

  test "transform" do
    x = [
      "",
      "test sentence",
      "another test sentence",
      "pi is 3.14159.",
    ]
    y = [
      [""],
      ["test", "sentence"],
      ["another", "test", "sentence"],
      ["pi", "is", "3", ".", "1", "4", "1", "5", "9", "."],
    ]
    assert PTBDigitTokenizer.transform(%{}, %{}, x) === y
    assert PTBDigitTokenizer.transform(%{}, %{}, y) === x
  end
end
