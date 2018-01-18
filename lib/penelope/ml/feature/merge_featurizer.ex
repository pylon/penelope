defmodule Penelope.ML.Feature.MergeFeaturizer do
  @moduledoc """
  This sequence featurizer invokes a set of inner featurizers and merges
  their results into a single map per sequence element.

  Example:
  ```
    features = [
      {:token_featurizer, []},
      {:regex_featurizer, [regexes: [~r/ed$/, ~r/ing$/]]},
    ]
    pipeline = [
      {:ptb_tokenizer, []},
      {:feature_merge, features},
      {:crf_tagger, []},
    ]
    Penelope.ml.pipeline.fit(%{}, x, y, pipeline)
  ```
  """

  alias Penelope.ML.Registry
  alias Penelope.ML.Pipeline

  @doc """
  fits each of the configured inner featurizers
  """
  @spec fit(
          context :: map,
          x :: [[any]],
          y :: [[any]],
          features :: [{String.t() | atom, any}]
        ) :: [{atom, any}]
  def fit(context, x, y, features) do
    Enum.map(features, &do_fit(&1, context, x, y))
  end

  defp do_fit({name, options}, context, x, y) do
    module = Registry.lookup(name)

    # fit this feature, if supported
    # otherwise, compile to an atom-keyed map
    model =
      Pipeline.call_maybe(module, :fit, [context, x, y, options], fn ->
        Map.new(options)
      end)

    {module, model}
  end

  @doc """
  transform a list of feature sequences using the inner featurizers and
  merge the results into a single list per sequence element
  """
  @spec transform(model :: [{atom, any}], context :: map, x :: [[any]]) :: [
          [map]
        ]
  def transform(model, context, x) do
    Enum.map(x, &do_transform(model, context, &1))
  end

  defp do_transform(model, context, x) do
    Enum.reduce(
      model,
      Enum.map(x, fn _x -> %{} end),
      &do_featurize(&1, context, &2, x)
    )
  end

  defp do_featurize({module, model}, context, r_x, x) do
    # call the inner featurizer
    [x] =
      Pipeline.call_maybe(module, :transform, [model, context, [x]], fn ->
        [nil]
      end)

    # if results were produced, merge them into the existing maps
    if x do
      Enum.map(Enum.zip(r_x, x), fn {r_x, x} -> Map.merge(r_x, x) end)
    else
      r_x
    end
  end

  @doc """
  imports parameters from a serialized model
  """
  @spec compile(params :: [map]) :: [{atom, any}]
  def compile(params) do
    Pipeline.compile(params)
  end

  @doc """
  exports a runtime model to a serializable data structure
  """
  @spec export(model :: [{atom, any}]) :: [map]
  def export(model) do
    Pipeline.export(model)
  end
end
