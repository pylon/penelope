defmodule Penelope.ML.Feature.ContextFeaturizer do
  @moduledoc """
  This is a sequence featurizer that extracts a constant value from the
  prediction context and adds it as a feature for each element in each
  sequence for each sample. This is useful for biasing a sequence
  classifier at the sample level.

  Example:
  ```
    model:   %{keys: ["k"]}
    context: %{k: 1.5}
    x:       [[1, 2, 3]]

    result:  [[%{k: 1.5}, %{k: 1.5}, %{k: 1.5}]]
  ```
  """

  @doc """
  transforms the sequence feature list
  """
  @spec transform(model::%{keys: [String.t]}, context::map, x::[[any]])
    :: [[map]]
  def transform(model, context, x) do
    features =
      model.keys
      |> Enum.map(&String.to_existing_atom/1)
      |> Map.new(fn k -> {k, context[k]} end)

    Enum.map(x, &Enum.map(&1, fn _x -> features end))
  end
end
