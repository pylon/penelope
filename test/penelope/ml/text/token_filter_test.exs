defmodule Penelope.ML.Text.TokenFilterTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.TokenFilter

  test "transform" do
    x = [
      [],
      ["test", "sentence"],
      ["another", "test", "Sentence"]
    ]

    stopwords = ["test", "sentence"]

    expect = [
      [],
      [],
      ["another", "Sentence"]
    ]

    assert TokenFilter.transform(%{stopwords: stopwords}, %{}, x) === expect
  end
end
