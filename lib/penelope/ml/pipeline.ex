defmodule Penelope.ML.Pipeline do
  @moduledoc """
  The ML pipeline provides the ability to express an inference graph as
  a data structure, and to fit/export/compile/predict based on the graph. A
  pipeline is represented as a sequence of stages, each of which is a
  component module that supports the pipeline interface. This structure is
  modeled after sklearn's pipeline.

  A pipeline component is either a transformer (supports the transform
  function) or a predictor (supports one or more predict functions).
  Components may optionally support the fit/export/compile functions. Below
  is a spec for each:

  ```elixir
  fit(context::map, x::[any], y::[any], options::keyword) :: any
  transform(model::any, context::map, x::[any]) :: [any]
  predict_class(model::any, context::map, x::[any]) :: [any]
  predict_probability(model::any, context::map, x::[any]) :: [%{any => float}]
  predict_sequence(model::any, context::map, x::[any]) :: [{[any], float}]
  export(model::any) :: map
  compile(params::map) :: any
  ```

  `fit` is used to train the model component and return its compiled model.
  `transform` transforms an incoming list of samples (feature matrix or list
  of sequences) for further pipeline processing. The `predict` functions
  output classes or sequences for a list of samples. `export` is used to
  serialize a model for persistance, and `compile` deserializes an exported
  model for inference.

  Compiled models are generally maps, but they can be any data structure.
  The `context` parameter is any user-supplied value, which an be used to
  thread a per-inference runtime parameter into a model (see
  the context_featurizer component for an example).

  Some components may not need custom fit/compile/export logic. For these
  components, the pipeline automatically compiles the fit options as a
  map.

  The pipeline uses the registry module for component name resolution. Names
  may be aliases or module atoms.

  The following is an example of a simple classification pipeline. It uses
  the token count vectorizer to count the total number of tokens in each
  sample string as a feature value.

  ```elixir
  pipeline = [
    {"ptb_tokenizer", []},
    {"count_vectorizer", []},
    {"svm_classifier", [kernel: :rbf, c: 2.0]}
  ]
  x_train = [
    "big daddy bear",
    "momma bear",
    "baby",
    "big bear daddy",
    "your momma",
    "lilbear"
  ]
  y_train = ["c", "b", "a", "c", "b", "a"]

  Penelope.ML.Pipeline.fit(%{}, x_train, y_train, pipeline)
  ```
  """

  alias Penelope.ML.Registry

  @doc """
  transforms and fits each stage of the pipeline

  A stage is a tuple of <name, options> where name is a registered name
  or module atom, and options are the parameters to the component's
  fit function.
  """
  @spec fit(
          context :: map,
          x :: [any],
          y :: [any],
          stages :: [{String.t() | atom, any}]
        ) :: [{atom, any}]
  def fit(context, x, y, stages) do
    {model, _x, _y} =
      Enum.reduce(
        stages,
        {[], x, y},
        &do_fit(&1, &2, context)
      )

    model
  end

  defp do_fit({name, options}, {result, x, y}, context) do
    module = Registry.lookup(name)

    # fit this stage, if supported
    # otherwise, compile to an atom-keyed map
    model =
      call_maybe(module, :fit, [context, x, y, options], fn ->
        Map.new(options)
      end)

    # transform the stage, if supported
    x = call_maybe(module, :transform, [model, context, x], fn -> x end)

    # save the compiled model and propagate the transformed x
    {result ++ [{module, model}], x, y}
  end

  @doc """
  transforms a list of samples through the pipeline
  """
  @spec transform(model :: [{atom, any}], context :: map, x :: [any]) :: [
          any
        ]
  def transform(model, context, x) do
    Enum.reduce(model, x, &do_transform(&1, context, &2))
  end

  defp do_transform({module, model}, context, x) do
    call_maybe(module, :transform, [model, context, x], fn -> x end)
  end

  @doc """
  class prediction

  This function predicts a list of classes (in the model) for each sample.
  """
  @spec predict_class(model :: [{atom, any}], context :: map, x :: [any]) ::
          [
            any
          ]
  def predict_class(model, context, x) do
    do_predict(model, context, x, :predict_class)
  end

  @doc """
  class probability prediction

  This function predicts the probability of each class (in a map) for
  each sample.
  """
  @spec predict_probability(
          model :: [{atom, any}],
          context :: map,
          x :: [any]
        ) :: [%{any => float}]
  def predict_probability(model, context, x) do
    do_predict(model, context, x, :predict_probability)
  end

  @doc """
  performs a sequence-to-sequence inference, returning the output
  sequences and sequence probabilities for each sample
  """
  @spec predict_sequence(
          model :: [{atom, any}],
          context :: map,
          x :: [[any]]
        ) :: [{[any], float}]
  def predict_sequence(model, context, x) do
    do_predict(model, context, x, :predict_sequence)
  end

  defp do_predict([{module, model}], context, x, method) do
    # at the end of the pipeline, so just call the prediction function
    apply(module, method, [model, context, x])
  end

  defp do_predict([{module, model} | models], context, x, method) do
    # in the middle of the pipeline, so call the transform function
    x = call_maybe(module, :transform, [model, context, x], fn -> x end)
    do_predict(models, context, x, method)
  end

  @doc """
  imports parameters from a serialized model
  """
  @spec compile(params :: [map]) :: [{atom, any}]
  def compile(params) do
    Enum.map(_stages = params, &do_compile/1)
  end

  defp do_compile(stage) do
    module = Registry.lookup(stage["name"])
    params = Map.get(stage, "params", %{})

    # perform custom compilation if supported
    # otherwise, compile to an atom-keyed map
    model =
      call_maybe(module, :compile, [params], fn ->
        Map.new(params, fn {k, v} -> {String.to_existing_atom(k), v} end)
      end)

    {module, model}
  end

  @doc """
  exports a runtime model to a serializable data structure
  """
  @spec export(model :: [{atom, any}]) :: [map]
  def export(model) do
    Enum.map(model, &do_export/1)
  end

  defp do_export({module, model}) do
    name = Registry.invert(module)

    # perform custom export if supported
    # otherwise, export to a string-keyed map
    params =
      call_maybe(module, :export, [model], fn ->
        Map.new(model, fn {k, v} -> {to_string(k), v} end)
      end)

    %{"name" => name, "params" => params}
  end

  @doc """
  calls a function on a module if it is supported, with a default fallback
  """
  @spec call_maybe(
          module :: atom,
          function :: atom,
          args :: [any],
          default :: function
        ) :: any
  def call_maybe(module, function, args, default) do
    if function_exported?(module, function, Enum.count(args)) do
      apply(module, function, args)
    else
      default.()
    end
  end
end
