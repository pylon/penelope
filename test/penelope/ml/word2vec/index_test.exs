defmodule Penelope.ML.Word2vec.IndexTest do
  @moduledoc """
  These tests verify the word2vec index module.
  """

  use ExUnit.Case, async: false

  alias Penelope.ML.Vector, as: Vector
  alias Penelope.ML.Word2vec.Index, as: Index
  alias Penelope.ML.Word2vec.IndexError, as: IndexError

  setup_all do
    input = "/tmp/penelope_ml_word2vec_index.txt"
    output = "/tmp/penelope_ml_word2vec_index"

    File.write!(
      input,
      1..10
      |> Enum.map(fn i ->
        "a" <>
          Integer.to_string(i) <>
          " " <>
          (1..10
           |> Enum.map(fn j -> Float.to_string(i / j) end)
           |> Enum.join(" "))
      end)
      |> Enum.join("\n")
    )

    on_exit(fn ->
      File.rm(input)
      File.rm_rf(output)
    end)

    {:ok, input: input, output: output}
  end

  test "parse" do
    assert_raise ArgumentError, fn ->
      Index.parse_line!("hello invalid")
    end

    assert Index.parse_line!("a 1") == {"a", Vector.from_list([1])}
    assert Index.parse_line!("b 1 2.5") == {"b", Vector.from_list([1, 2.5])}
  end

  test "build", %{input: input, output: output} do
    index = Index.create!(output, "test", vector_size: 10)

    assert_raise IndexError, fn ->
      {term, vector} = Index.parse_line!("invalid 1")
      Index.insert!(index, {term, 1, vector})
    end

    assert_raise IndexError, fn ->
      Index.parse_insert!(index, {"invalid 1", 1})
    end

    Index.compile!(index, input)
    {term, vector} = Index.parse_line!("hello 1 2 3 4 5 6 7 8 9 10")
    Index.insert!(index, {term, 11, vector})
    Index.parse_insert!(index, {"goodbye 10 9 8 7 6 5 4 3 2 1", 12})

    Index.close(index)

    index = Index.open!(output)

    assert Index.fetch!(index, 0) == nil
    assert Index.fetch!(index, 11) == "hello"
    assert Index.fetch!(index, 12) == "goodbye"

    assert Index.lookup!(index, "invalid") == {0, Vector.zeros(10)}
    assert Index.lookup!(index, "missing") == {0, Vector.zeros(10)}
    assert Index.lookup!(index, "hello") == {11, Vector.from_list(1..10)}
    assert Index.lookup!(index, "goodbye") == {12, Vector.from_list(10..1)}

    1..10
    |> Enum.each(fn i ->
      word = "a" <> Integer.to_string(i)

      vector =
        1..10
        |> Enum.map(fn j -> i / j end)
        |> Vector.from_list()

      assert Index.fetch!(index, i) == word
      assert Index.lookup!(index, word) == {i, vector}
    end)

    Index.close(index)
  end
end
