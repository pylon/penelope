defmodule Mix.Tasks.Word2vec.CompileTest do

  use ExUnit.Case, async: true

  alias Mix.Tasks.Word2vec.Compile

  setup_all do
    input  = "/tmp/penelope_mix_tasks_word2vec_compile.txt"
    output = "/tmp/penelope_mix_tasks_word2vec_compile"

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

  test "index construction", %{input: input, output: output} do
    Compile.run([
      input,
      output,
      "test",
      "--partitions=3",
      "--size-hint=1000",
      "--vector-size=10",
    ])
  end
end
