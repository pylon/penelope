defmodule Penelope.ML.Word2vec.MeanVectorizerTest do
  @moduledoc """
  These tests verify the word2vec mean vectorizer.
  """

  use ExUnit.Case, async: false

  alias Penelope.ML.Vector, as: Vector
  alias Penelope.ML.Word2vec.Index, as: Index
  alias Penelope.ML.Word2vec.MeanVectorizer, as: Vectorizer

  setup_all do
    output = "/tmp/penelope_ml_word2vec_meanvectorizertest"

    index = Index.create!(output, "test", vector_size: 2)
    Index.parse_insert!(index, {"the 1 1.5", 1})
    Index.parse_insert!(index, {"fox 2 4.5", 2})
    Index.parse_insert!(index, {"dog 5 7.5", 3})
    Index.parse_insert!(index, {"horse 0 3", 4})

    on_exit(fn ->
      Index.close(index)
      File.rm_rf(output)
    end)

    {:ok, word2vec_index: index}
  end

  test "transform", context do
    x = [
      ["the", "quick", "brown", "fox"],
      ["the", "lazy", "dog"],
      ["some", "old", "horse"]
    ]

    expect = [
      Vector.from_list([0.75, 1.5]),
      Vector.from_list([2.0, 3.0]),
      Vector.from_list([0.0, 1.0])
    ]

    assert Vectorizer.transform(%{}, context, x) === expect
  end
end
