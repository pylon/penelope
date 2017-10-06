defmodule AssertionTest do
  @moduledoc """
  These tests verify the word2vec index module.
  """

  use ExUnit.Case, async: true

  alias NLPotion.ML.Word2vec.Index, as: Index
  alias NLPotion.ML.Word2vec.IndexError, as: IndexError

  setup_all do
    input  = "/tmp/penelope_word2vec.txt"
    output = "/tmp/penelope_word2vec"

    File.write!(
      input,
      1..10
      |> Enum.map(fn i -> "a" <> Integer.to_string(i) <> " " <>
                          (1..10
                           |> Enum.map(fn j -> Float.to_string(i / j) end)
                           |> Enum.join(" ")) end)
      |> Enum.join("\n"))

    on_exit fn ->
      File.rm(input)
      File.rm_rf(output)
    end

    {:ok, input: input, output: output}
  end

  test "parse" do
    assert_raise ArgumentError, fn ->
      Index.parse_line!("hello invalid")
    end

    assert Index.parse_line!("a 1") == {"a", make_vector([1])}
    assert Index.parse_line!("b 1 2.5") == {"b", make_vector([1, 2.5])}
  end

  test "build", %{input: input, output: output} do
    index = Index.create!(output, "index_test", vector_size: 10)

    assert_raise IndexError, fn ->
      Index.insert!(index, Index.parse_line!("invalid 1"))
    end
    assert_raise IndexError, fn ->
      Index.parse_insert!(index, "invalid 1")
    end

    Index.insert!(index, Index.parse_line!("hello 1 2 3 4 5 6 7 8 9 10"))
    Index.parse_insert!(index, "goodbye 10 9 8 7 6 5 4 3 2 1")

    Index.compile!(index, input)
    Index.close(index)

    index = Index.open!(output)
    assert Index.lookup!(index, "invalid") == make_vector(for _ <- 1..10, do: 0)
    assert Index.lookup!(index, "missing") == make_vector(for _ <- 1..10, do: 0)
    assert Index.lookup!(index, "hello") == make_vector(1..10)
    assert Index.lookup!(index, "goodbye") == make_vector(10..1)

    1..10
    |> Enum.each(fn i ->
        word   = "a" <> Integer.to_string(i)
        vector = 1..10
                 |> Enum.map(fn j -> i / j end)
                 |> make_vector()
        assert(Index.lookup!(index, word) == vector)
      end)

    Index.close(index)
  end

  defp make_vector(numbers) do
    numbers
    |> Enum.map(&<<&1::float-little-size(32)>>)
    |> Enum.reduce(&(&2 <> &1))
  end
end
