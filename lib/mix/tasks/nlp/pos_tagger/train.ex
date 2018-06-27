defmodule Mix.Tasks.Nlp.PosTagger.Train do
  @moduledoc """
  This task trains and optionally tests a part-of-speech tagger
  using files containing tokenized text and POS tags.

  The trained model will be JSON-encoded and saved to
  the specified target file; it can be loaded by reading
  the target file, JSON decoding it, and feeding the result to
  `POSTagger.compile/1`.

  If supplied, the test file must be formatted identically to the training file.
  """
  @shortdoc @moduledoc

  use Mix.Task

  alias Penelope.NLP.POSTagger
  alias Mix.Tasks.Nlp.PosTagger.TaggerUtils

  require Logger

  @switches [
    section_sep: :string,
    token_sep: :string,
    test_file: :string
  ]

  def run(argv) do
    {options, args, other} = OptionParser.parse(argv, switches: @switches)

    case {args, other} do
      {[source, target], []} -> execute(source, target, options)
      _ -> usage()
    end
  end

  defp execute(source, target, options) do
    Application.ensure_all_started(:penelope)
    options = combine_with_defaults(options)
    {tokens, tags} = TaggerUtils.ingest_file(source, options)
    Logger.info("Training file processed; fitting model...")
    model = POSTagger.fit(%{}, tokens, tags)

    exported =
      model
      |> POSTagger.export()
      |> Poison.encode!()

    File.write!(target, exported)
    Logger.info("Model saved to #{target}.")

    if options.test_file do
      test(model, options)
    end
  end

  defp usage do
    IO.puts("""
    Part-of-speech tagger trainer
    usage: mix nlp.pos_tagger.train [options] <train-file> <output-file>

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

  defp test(model, options) do
    {tokens, tags} = TaggerUtils.ingest_file(options.test_file, options)
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
      section_sep: Keyword.get(options, :section_sep, "\t"),
      test_file: Keyword.get(options, :test_file)
    }
  end
end
