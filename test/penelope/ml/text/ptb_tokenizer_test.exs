defmodule Penelope.ML.Text.PTBTokenizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.PTBTokenizer

  test "transform" do
    x = ["", "test sentence", "another test sentence"]
    y = [[""], ["test", "sentence"], ["another", "test", "sentence"]]
    assert PTBTokenizer.transform(%{}, %{}, x) === y
    assert PTBTokenizer.transform(%{}, %{}, y) === x
  end
end
