defmodule Penelope.ML.Text.CountVectorizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.CountVectorizer
  alias Penelope.ML.Vector

  test "transform" do
    x = [
      [],
      ["test", "sentence"],
      ["another", "test", "sentence"]
    ]

    expect = [
      Vector.zeros(1),
      Vector.from_list([2]),
      Vector.from_list([3])
    ]

    assert CountVectorizer.transform(%{}, %{}, x) === expect
  end
end
