defmodule Penelope.ML.Text.RegexVectorizer do
  @moduledoc """
  The regex vectorizer applies a list of regexes to each incoming document
  and produces an output vector of 0/1 values based on the results.
  """

  alias Penelope.ML.Vector

  @doc """
  transforms a list of documents into vectors
  """
  @spec transform(model::%{regexes: [String.t]}, context::map, x::[String.t])
    :: [Vector.t]
  def transform(model, _context, x) do
    Enum.map(x, &match_regexes(&1, model.regexes))
  end

  defp match_regexes(x, regexes) do
    regexes
    |> Enum.map(&match_regex(x, &1))
    |> Vector.from_list()
  end

  defp match_regex(x, regex) do
    Regex.run(regex, x) && 1.0 || 0.0
  end
end
