defmodule Penelope.ML.CRF.Tagger do
  @moduledoc """
  The CRF tagger is a thin wrapper over the CRFSuite library for sequence
  inference. It provides the ability to train sequence models, use them
  for inference, and import/export them.

  Features (Xs) are represented as lists of sequences (lists). Each sequence
  entry can contain a string (for simple word-based features), a list of
  stringable values (list features), or maps (for named features per sequence
  item).

  Labels (Ys) are represented as lists of sequences of strings. Each label
  must correspond to an entry in the feature lists.

  Models are compiled/exported to/from a map containing a binary blob
  that is maintained by CRF suite. Training parameters are analogs of those
  used by the sklearn-crfsuite library. For more information, see:
    http://www.chokkan.org/software/crfsuite/
    https://sklearn-crfsuite.readthedocs.io/en/latest/
  """
  alias Penelope.NIF

  @doc """
  trains a CRF model and adds it to the pipeline context map

  options:
  |key                       |default             |
  |--------------------------|--------------------|
  |`algorithm`               |`:lbfgs`            |
  |`min_freq`                |0.0                 |
  |`all_possible_states`     |false               |
  |`all_possible_transitions`|false               |
  |`c1`                      |0.0                 |
  |`c2`                      |0.0                 |
  |`max_iterations`          |depends on algorithm|
  |`num_memories`            |6                   |
  |`epsilon`                 |1e-5                |
  |`period`                  |10                  |
  |`delta`                   |1e-5                |
  |`linesearch`              |:more_thuente       |
  |`max_linesearch`          |20                  |
  |`calibration_eta`         |0.1                 |
  |`calibration_rate`        |2.0                 |
  |`calibration_samples`     |1000                |
  |`calibration_candidates`  |10                  |
  |`calibration_max_trials`  |20                  |
  |`pa_type`                 |1                   |
  |`c`                       |1.0                 |
  |`error_sensitive`         |true                |
  |`averaging`               |true                |
  |`variance`                |1.0                 |
  |`gamma`                   |1.0                 |

  algorithms:
  `:lbfgs`, `:l2sgd`, `:ap`, `:pa`, `:arow`

  linesearch:
  `:more_thuente`, `:backtracking`, `:strong_backtracking`

  for more information on parameters, see
    https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
  """
  @spec fit(
    context::map,
    x::[[String.t | list | map]],
    y::[[String.t]], options::keyword
  ) :: {map, [[map]], [[String.t]]}
  def fit(context, x, y, options \\ []) do
    if length(x) !== length(y), do: raise ArgumentError, "mismatched x/y"

    x = Enum.map(x, fn x -> Enum.map(x, &featurize/1) end)
    params = fit_params(x, y, options)
    model = NIF.crf_train(x, y, params)

    {Map.put(context, :crf_model, model), x, y}
  end

  @doc """
  extracts model parameters from the pipeline context map

  These parameters are simple elixir objects and can later be passed to
  `compile` to add the model back to the context.
  """
  @spec export(%{crf_model: reference}) :: map
  def export(%{crf_model: crf_model}) do
    NIF.crf_export(crf_model)
  end

  @doc """
  compiles a pre-trained model and adds it to the pipeline context
  """
  @spec compile(context::map, params::map) :: map
  def compile(context, params) do
    model = NIF.crf_compile(params)
    Map.put(context, :crf_model, model)
  end

  @doc """
  predicts a target sequence from a feature sequence
  returns the predicted sequence and its probability
  """
  @spec predict(%{crf_model: reference}, x::[[String.t | list | map]])
    :: {[String.t], float}
  def predict(%{crf_model: model}, x) do
    x = Enum.map(x, &featurize/1)
    NIF.crf_predict(model, x)
  end

  defp fit_params(_x, _y, options) do
    algorithm = Keyword.get(options, :algorithm, :lbfgs)
    min_freq = Keyword.get(options, :min_freq, 0) / 1
    all_states? = Keyword.get(options, :all_possible_states?, false)
    all_transitions? = Keyword.get(options, :all_possible_transitions?, false)
    c1 = Keyword.get(options, :c1, 0.0) / 1
    c2 = Keyword.get(options, :c2, 0.0) / 1
    max_iter = Keyword.get(options, :max_iterations, max_iterations(algorithm))
    num_memories = Keyword.get(options, :num_memories, 6)
    epsilon = Keyword.get(options, :epsilon, 1.0e-5) / 1
    period = Keyword.get(options, :period, 10)
    delta = Keyword.get(options, :delta, 1.0e-5) / 1
    linesearch = Keyword.get(options, :linesearch, :more_thuente)
    max_linesearch = Keyword.get(options, :max_linesearch, 20)
    calibration_eta = Keyword.get(options, :calibration_eta, 0.1) / 1
    calibration_rate = Keyword.get(options, :calibration_rate, 2.0) / 1
    calibration_samples = Keyword.get(options, :calibration_samples, 1000)
    calibration_candidates = Keyword.get(options, :calibration_candidates, 10)
    calibration_max_trials = Keyword.get(options, :calibration_max_trials, 20)
    pa_type = Keyword.get(options, :pa_type, 1)
    c = Keyword.get(options, :c, 1.0) / 1
    error_sensitive? = Keyword.get(options, :error_sensitive?, true)
    averaging? = Keyword.get(options, :averaging?, true)
    variance = Keyword.get(options, :variance, 1.0) / 1
    gamma = Keyword.get(options, :gamma, 1.0) / 1

    %{
      algorithm:                 algorithm,
      min_freq:                  min_freq,
      all_possible_states?:      all_states?,
      all_possible_transitions?: all_transitions?,
      c1:                        c1,
      c2:                        c2,
      max_iterations:            max_iter,
      num_memories:              num_memories,
      epsilon:                   epsilon,
      period:                    period,
      delta:                     delta,
      linesearch:                linesearch_param(linesearch),
      max_linesearch:            max_linesearch,
      calibration_eta:           calibration_eta,
      calibration_rate:          calibration_rate,
      calibration_samples:       calibration_samples,
      calibration_candidates:    calibration_candidates,
      calibration_max_trials:    calibration_max_trials,
      pa_type:                   pa_type,
      c:                         c,
      error_sensitive?:          error_sensitive?,
      averaging?:                averaging?,
      variance:                  variance,
      gamma:                     gamma,
    }
  end

  defp max_iterations(algorithm) do
    case algorithm do
      :lbfgs -> 2_147_483_647
      :l2sgd -> 1000
      :ap    -> 100
      :pa    -> 100
      :arow  -> 100
    end
  end

  defp linesearch_param(linesearch) do
    case linesearch do
      :more_thuente        -> :"MoreThuente"
      :backtracking        -> :"Backtracking"
      :strong_backtracking -> :"StrongBacktracking"
    end
  end

  # convert from a set of feature formats to the standard crfsuite format:
  # [%{"feature1" => 1.0, "feature2" => 1.5, ...}, ...]
  # . simple tokens:
  #   "t1" -> %{"t1" => 1.0}
  # . list per token:
  #   ["f1", "f2", ...] -> %{"f1" => 1.0, "f2" => 1.0, ...}
  # . map per token:
  #   %{"f1" => "v1", "f2" -> 1.5, ...} -> %{"f1-v1" => 1.0, "f2" => 1.5, ...}
  defp featurize(x) when is_map(x) do
    Map.new(x, fn {k, v} -> hd(Map.to_list(featurize(k, v))) end)
  end
  defp featurize(x) when is_list(x) do
    Map.new(x, fn v -> hd(Map.to_list(featurize(v, 1))) end)
  end
  defp featurize(x) do
    featurize(x, 1)
  end
  defp featurize(k, v) when is_number(v) do
    %{to_string(k) => v / 1}
  end
  defp featurize(k, v) do
    featurize("#{k}-#{v}", 1)
  end
end
