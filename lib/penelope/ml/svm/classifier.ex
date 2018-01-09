defmodule Penelope.ML.SVM.Classifier do
  @moduledoc """
  The SVM classifier uses libsvm for multi-class classification. It provides
  support for training a model, compiling/extracting model parameters to/from
  erlang data structures, and predicting classes or probabilities.

  Features are represented as lists of dense Vector instances. Classes can
  be any value, and class labels for training are lists of these.

  Model parameters are elixir analogs of those supported by libsvm. See
  https://github.com/cjlin1/libsvm for details.
  """

  alias Penelope.ML.Vector
  alias Penelope.NIF

  @doc """
  trains an SVM model and returns it as a compiled model

  |key           |description                               |default  |
  |--------------|------------------------------------------|---------|
  |`kernel`      |one of `:linear`/`:rbf`/`:poly`/`:sigmoid`|`:linear`|
  |`degree`      |polynomial degree                         |3        |
  |`gamma`       |training example reach - `:auto` for 1/N  |`:auto`  |
  |`coef0`       |independent term                          |0.0      |
  |`c`           |error term penalty                        |1.0      |
  |`weights`     |class weights map - `:auto` for balanced  |`:auto`  |
  |`epsilon`     |tolerance for stopping                    |0.001    |
  |`cache_size`  |kernel cache size, in MB                  |1        |
  |`shrinking?`  |use the shrinking heuristic?              |true     |
  |`probability?`|enable class probabilities?               |false    |
  """
  @spec fit(context::map, x::[Vector.t], y::[any], options::keyword) :: map
  def fit(_context, x, y, options \\ []) do
    if length(x) !== length(y), do: raise ArgumentError, "mismatched x/y"

    classes = Enum.uniq(y)
    y = Enum.map(y, &index_of(classes, &1))

    params = fit_params(x, y, classes, options)
    model = NIF.svm_train(x, y, params)
    %{svm: model, classes: classes}
  end

  @doc """
  extracts model parameters from the compiled model

  These parameters are simple elixir objects and can later be passed to
  `compile` to prepare the model for inference.
  """
  @spec export(%{svm: reference, classes: [any]}) :: map
  def export(%{svm: model, classes: classes}) do
    model
    |> NIF.svm_export()
    |> Map.put(:classes, classes)
    |> Map.update!(:kernel, &to_string/1)
    |> Map.update!(:coef, fn l -> Enum.map(l, &Vector.to_list/1) end)
    |> Map.update!(:sv, fn l -> Enum.map(l, &Vector.to_list/1) end)
    |> Map.update!(:rho, &Vector.to_list/1)
    |> Map.update!(:prob_a, fn v -> v && Vector.to_list(v) end)
    |> Map.update!(:prob_b, fn v -> v && Vector.to_list(v) end)
    |> Map.new(fn {k, v} -> {to_string(k), v} end)
  end

  @doc """
  compiles a pre-trained model
  """
  @spec compile(params::map) :: map
  def compile(%{"classes" => classes} = params) do
    model =
      params
      |> Map.new(fn {k, v} -> {String.to_existing_atom(k), v} end)
      |> Map.put(:classes, Enum.to_list(0..length(classes) - 1))
      |> Map.update!(:kernel, &String.to_existing_atom/1)
      |> Map.update!(:coef, fn l -> Enum.map(l, &Vector.from_list/1) end)
      |> Map.update!(:sv, fn l -> Enum.map(l, &Vector.from_list/1) end)
      |> Map.update!(:rho, &Vector.from_list/1)
      |> Map.update!(:prob_a, fn v -> v && Vector.from_list(v) end)
      |> Map.update!(:prob_b, fn v -> v && Vector.from_list(v) end)
      |> NIF.svm_compile()

    %{svm: model, classes: classes}
  end

  @doc """
  predicts a list of target classes from a list of feature vectors
  """
  @spec predict_class(
    %{svm: reference, classes: [any]},
    context::map,
    [x::Vector.t]
  ) :: [any]
  def predict_class(model, _context, x) do
    Enum.map(x, &do_predict_class(model, &1))
  end

  defp do_predict_class(%{svm: model, classes: classes}, x) do
    Enum.at(classes, NIF.svm_predict_class(model, x))
  end

  @doc """
  predicts probabilities for all classes from a feature vector

  The results are returned in a map of `%{label => probability}`.
  """
  @spec predict_probability(
    %{svm: reference, classes: [any]},
    context::map,
    [x::Vector.t]
  ) :: [%{any => float}]
  def predict_probability(model, _context, x) do
    Enum.map(x, &do_predict_probability(model, &1))
  end

  defp do_predict_probability(%{svm: model, classes: classes}, x) do
    model
    |> NIF.svm_predict_probability(x)
    |> Map.new(fn {k, p} -> {Enum.at(classes, k), p} end)
  end

  defp fit_params(x, y, classes, options) do
    gamma = with :auto <- Keyword.get(options, :gamma, :auto) do
      auto_gamma(x)
    end

    weights = case Keyword.get(options, :weights, :auto) do
      :auto   -> auto_weights(y)
      weights -> manual_weights(classes, weights)
    end

    %{
      kernel:       Keyword.get(options, :kernel, :linear),
      degree:       Keyword.get(options, :degree, 3),
      gamma:        gamma / 1,
      coef0:        Keyword.get(options, :coef0, 0) / 1,
      c:            Keyword.get(options, :c, 1) / 1,
      weights:      weights,
      epsilon:      Keyword.get(options, :epsilon, 1.0e-3) / 1,
      cache_size:   Keyword.get(options, :cache_size, 1) / 1,
      shrinking?:   Keyword.get(options, :shrinking?, true),
      probability?: Keyword.get(options, :probability?, false)
    }
  end

  defp auto_gamma([x | _]) do
    1.0 / Vector.size(x)
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
