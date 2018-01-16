defmodule Penelope.ML.Text.LowercasePreprocessorTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.LowercasePreprocessor

  test "transform" do
    x = ["", "TeSt sEnTeNcE", "another test sentence"]
    y = ["", "test sentence", "another test sentence"]
    assert LowercasePreprocessor.transform(%{}, %{}, x) === y
    assert LowercasePreprocessor.transform(%{}, %{}, y) === y
  end
end
