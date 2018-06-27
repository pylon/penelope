defmodule Mix.Tasks.Nlp.PosTagger.Test do
  @moduledoc """
  This task tests a pretrained part-of-speech tagger model
  using a file containing tokenized text and POS tags.
  """
  @shortdoc @moduledoc

  use Mix.Task

  alias Penelope.NLP.POSTagger
  alias Mix.Tasks.Nlp.PosTagger.TaggerUtils

  require Logger

  @switches [
    section_sep: :string,
    token_sep: :string
  ]

  def run(argv) do
    {options, args, other} = OptionParser.parse(argv, switches: @switches)

    case {args, other} do
      {[source, target], []} -> execute(source, target, options)
      _ -> usage()
    end
  end

  defp execute(model_file, test_file, options) do
    Application.ensure_all_started(:penelope)

    model_file
    |> File.read!()
    |> Poison.decode!()
    |> POSTagger.compile()
    |> test(test_file, options)
  end

  defp usage do
    IO.puts("""
    Part-of-speech tagger tester
    usage: mix nlp.pos_tagger.test [options] <model-file> <test-file>

    train-file:  path to a training file, each line of which contains a
                 tokenized phrase and the tokens' part-of-speech tags
                 Example line: Bill|saw|her,NNP|VBD|PRP
    output-file: path to the file where the trained model should be saved

    options:
    --section-sep: the string separating tokenized text from POS tags in
                   each line of the training file, default: "\\t"
    --token-sep:   the string separating individual tokens and tags in each
                   line of the training file, default: " "
    --test-file:   path to the file to use for testing the trained tagger.
                   must be formatted identically to the training file,
                   default: nil
    """)
  end

  defp test(model, test_file, options) do
    options = combine_with_defaults(options)
    {tokens, tags} = TaggerUtils.ingest_file(test_file, options)
    tags = List.flatten(tags)
    Logger.info("Test file loaded. Testing tagger...")

    stats =
      tokens
      |> Enum.map(&POSTagger.tag(model, %{}, &1))
      |> Enum.flat_map(fn results -> Enum.map(results, &elem(&1, 1)) end)
      |> Enum.zip(tags)
      |> Enum.reduce({0, 0}, fn {predicted, gold}, {correct, total} ->
        {(predicted == gold && correct + 1) || correct, total + 1}
      end)

    stats = %{
      total_tokens: elem(stats, 1),
      accuracy: elem(stats, 0) / elem(stats, 1)
    }

    Logger.info(
      ~s(Test complete.) <>
        ~s(\n\tTotal tokens: #{stats.total_tokens}) <>
        ~s(\n\tAccuracy: #{stats.accuracy})
    )
  end

  defp combine_with_defaults(options) do
    %{
      token_sep: Keyword.get(options, :token_sep, " "),
      section_sep: Keyword.get(options, :section_sep, "\t")
    }
  end
end
