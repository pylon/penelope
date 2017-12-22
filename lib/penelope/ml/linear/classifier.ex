defmodule Penelope.ML.Linear.Classifier do
  @moduledoc """
  The linear classifier uses liblinear for multi-class classification. It
  provides support for training a model, compiling/extracting model
  parameters to/from erlang data structures, and predicting classes or
  probabilities.

  Features are represented as lists of dense Vector instances. Classes can
  be any value, and class labels for training are lists of these.

  Model parameters are elixir analogs of those supported by liblinear. See
  https://github.com/cjlin1/liblinear for details.
  """

  alias Penelope.ML.Vector
  alias Penelope.NIF

  @doc """
  trains a linear model and adds it to the pipeline context map

  ### options:
  |key           |description                              |default          |
  |--------------|-----------------------------------------|-----------------|
  |`solver`      |see solver types below                   |`:l2r_l2loss_svc`|
  |`c`           |error term penalty                       |1.0              |
  |`weights`     |class weights map, `:auto` for balanced  |`:auto`          |
  |`epsilon`     |tolerance for stopping                   |0.001            |
  |`bias`        |intercept bias (-1 for no intercept)     |1.0              |

  ### solver types
  |type                  |description                      |
  |----------------------|---------------------------------|
  |`:l2r_lr`             |primal L2 reg logistic regression|
  |`:l2r_l2loss_svc_dual`|dual L2 reg L2 loss SVC          |
  |`:l2r_l2loss_svc`     |primal L2 reg L2 loss SVC        |
  |`:l2r_l1loss_svc_dual`|dual L2 reg L1 loss SVC          |
  |`:mcsvm_cs`           |crammer/singer SVC               |
  |`:l1r_l2loss_svc`     |L1 reg L2 loss SVC               |
  |`:l1r_lr`             |L1 reg logistic regression       |
  |`:l2r_lr_dual`        |dual L2 reg logistic regression  |

  """
  @spec fit(context::map, x::[Vector.t], y::[any], options::keyword)
    :: {map, [Vector.t], [integer]}
  def fit(context, x, y, options \\ []) do
    if length(x) !== length(y), do: raise ArgumentError, "mismatched x/y"

    classes = Enum.uniq(y)
    y = Enum.map(y, &index_of(classes, &1))

    params = fit_params(x, y, classes, options)
    model = NIF.lin_train(x, y, params)
    context =
      context
      |> Map.put(:lin_model, model)
      |> Map.put(:lin_classes, classes)

    {context, x, y}
  end

  @doc """
  extracts model parameters from the pipeline context map

  These parameters are simple elixir objects and can later be passed to
  `compile` to add the model back to the context.
  """
  @spec export(%{lin_model: reference, lin_classes: [any]}) :: map
  def export(%{lin_model: model, lin_classes: classes}) do
    model
    |> NIF.lin_export()
    |> Map.put(:classes, classes)
    |> Map.update!(:coef, fn l -> Enum.map(l, &Vector.to_list/1) end)
    |> Map.update!(:intercept, fn
          l when is_binary(l) -> Vector.to_list(l)
          x -> x
       end)
  end

  @doc """
  compiles a pre-trained model and adds it to the pipeline context
  """
  @spec compile(context::map, params::map) :: map
  def compile(context, %{classes: classes} = params) do
    model =
      params
      |> Map.put(:classes, Enum.to_list(0..length(classes) - 1))
      |> Map.update!(:coef, fn l -> Enum.map(l, &Vector.from_list/1) end)
      |> Map.update!(:intercept, fn
            l when is_list(l) -> Vector.from_list(l)
            x -> x
         end)
      |> NIF.lin_compile()

    context
    |> Map.put(:lin_model, model)
    |> Map.put(:lin_classes, classes)
  end

  @doc """
  predicts a target class from a feature vector
  """
  @spec predict_class(
    %{lin_model: reference, lin_classes: [any]},
    x::Vector.t
  ) :: any
  def predict_class(%{lin_model: model, lin_classes: classes}, x) do
    Enum.at(classes, NIF.lin_predict_class(model, x))
  end

  @doc """
  predicts probabilities for all classes from a feature vector

  The results are returned in a map of `%{label => probability}`.
  """
  @spec predict_probability(
    %{lin_model: reference, lin_classes: [any]},
    x::Vector.t
  ) :: %{any => float}
  def predict_probability(%{lin_model: model, lin_classes: classes}, x) do
    model
    |> NIF.lin_predict_probability(x)
    |> Map.new(fn {k, p} -> {Enum.at(classes, k), p} end)
  end

  defp fit_params(_x, y, classes, options) do
    weights = case Keyword.get(options, :weights, :auto) do
      :auto   -> auto_weights(y)
      weights -> manual_weights(classes, weights)
    end

    %{
      solver:  Keyword.get(options, :solver, :l2r_l2loss_svc),
      c:       Keyword.get(options, :c, 1) / 1,
      weights: weights,
      epsilon: Keyword.get(options, :epsilon, 1.0e-4) / 1,
      p:       0.0,
      bias:    Keyword.get(options, :bias, 1) / 1
    }
  end

  defp auto_weights(y) do
    # class frequencies, sample count, and class count
    f = Enum.reduce(y, %{}, &Map.update(&2, &1, 1, fn f -> f + 1 end))
    m = Enum.sum(Map.values(f))
    k = Enum.count(f)

    # weight = samples / (classes * frequency)
    f
    |> Map.keys()
    |> Enum.reduce(f, &Map.update!(&2, &1, fn f -> m / (k * f) end))
  end

  defp manual_weights(classes, weights) do
    Map.new(weights, fn {k, v} -> {index_of(classes, k), v} end)
  end

  defp index_of(l, e) do
    Enum.find_index(l, fn x -> x === e end)
  end
end
