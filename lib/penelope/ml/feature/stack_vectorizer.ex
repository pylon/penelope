defmodule Penelope.ML.Feature.StackVectorizer do
  @moduledoc """
  This vectorizer horizontally stacks the results of a sequence of
  inner vectorizers applied to an incoming feature matrix. This is analogous
  to the behavior of the `FeatureUnion` component in sklearn.

  Example:
  ```
    features = [
      {:count_vectorizer, []},
      {:regex_vectorizer, [regexes: [~r/ed$/, ~r/ing$/]]},
    ]
    pipeline = [
      {:ptb_tokenizer, []},
      {:feature_stack, features},
      {:svm_classifier, [c: 2.0]},
    ]
    Penelope.ml.pipeline.fit(%{}, x, y, pipeline)
  ```
  """

  alias Penelope.ML.Vector
  alias Penelope.ML.Registry
  alias Penelope.ML.Pipeline

  @doc """
  fits each of the configured inner vectorizers
  """
  @spec fit(
    context::map,
    x::[any],
    y::[any],
    features::[{String.t | atom, any}]
  ) :: [{atom, any}]
  def fit(context, x, y, features) do
    Enum.map(features, &do_fit(&1, context, x, y))
  end

  defp do_fit({name, options}, context, x, y) do
    module = Registry.lookup(name)

    # fit this feature, if supported
    # otherwise, compile to an atom-keyed map
    model = Pipeline.call_maybe(
      module,
      :fit,
      [context, x, y, options],
      fn -> Map.new(options) end
    )

    {module, model}
  end

  @doc """
  transform a list of feature vectors using the inner featurizers and
  stack the results into a single vector per sample
  """
  @spec transform(model::[{atom, any}], context::map, x::[any])
    :: [Vector.t]
  def transform(model, context, x) do
    Enum.reduce(
      model,
      Enum.map(x, fn _x -> Vector.empty() end),
      &do_transform(&1, context, &2, x)
    )
  end

  defp do_transform({module, model}, context, r_x, x) do
    # call the inner vectorizer
    x = Pipeline.call_maybe(
      module,
      :transform,
      [model, context, x],
      fn -> nil end
    )

    # if results were produced, stack them into the existing vectors
    if x do
      Enum.map(Enum.zip(r_x, x), fn {r_x, x} -> Vector.concat(r_x, x) end)
    else
      r_x
    end
  end

  @doc """
  imports parameters from a serialized model
  """
  @spec compile(params::[map]) :: [{atom, any}]
  def compile(params) do
    Pipeline.compile(params)
  end

  @doc """
  exports a runtime model to a serializable data structure
  """
  @spec export(model::[{atom, any}]) :: [map]
  def export(model) do
    Pipeline.export(model)
  end
end
