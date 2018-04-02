defmodule Penelope.ML.Text.TokenFilterTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.TokenFilter

  test "transform" do
    x = [
      [],
      ["test", "sentence"],
      ["another", "test", "Sentence"]
    ]

    tokens = ["test", "sentence"]

    expect = [
      [],
      [],
      ["another", "Sentence"]
    ]

    assert TokenFilter.transform(%{tokens: tokens}, %{}, x) === expect
  end
end
