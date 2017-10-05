defmodule Mix.Tasks.Word2vec.Compile do
  @moduledoc """
  This task compiles a word vector text file into a set of DETS indexes.
  """
  @shortdoc @moduledoc

  use Mix.Task

  alias NLPotion.ML.Word2vec.Index, as: Index

  @switches [partitions: :integer, size_hint:  :integer]
  @defaults [partitions: 1, size_hint: 200_000]

  def run(argv) do
    case OptionParser.parse(argv, switches: @switches) do
      {[partitions: _, size_hint: _] = options,
       [source, target, name],
       []} -> execute(source, target, name, options)
      _ -> usage()
    end
  end

  defp execute(source, target, name, options) do
    options = Keyword.merge @defaults, options
    index = Index.create(target, name, options)
    try do
      Index.compile(index, source)
    after
      Index.close index
    end
  end

  defp usage do
    IO.puts """
      Word2Vec DETS Compiler
      usage: mix word2vec.compile [options] <source-file> <target-path> <name>

      source-file: path to a word2vec standard text file
      target-path: path to the output directory
      name:        name of the index to create in the output directory

      options:
        --partitions: number of partitions (files) to create, default: 1
        --size_hint:  index hint for the total number of words
    """
  end
end
