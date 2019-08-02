defmodule Penelope.ML.CRF.TaggerTest do
  @moduledoc """
  These tests verify the CRF tagger.

  Stress tests are disabled by default, but they can be used to detect
  memory leaks in the CRF NIF.
  """

  use ExUnit.Case, async: true

  import ExUnitProperties
  import Penelope.TestUtility

  alias Penelope.ML.CRF.Tagger
  alias StreamData, as: Gen

  @x_train [
    [],
    ["you", "have", "four", "pears"],
    ["these", "one", "hundred", "apples"]
  ]
  @y_train [
    [],
    ["o", "o", "b_num", "b_fruit"],
    ["o", "b_num", "i_num", "b_fruit"]
  ]

  test "fit/export/compile" do
    assert_raise(fn ->
      Tagger.fit(%{}, [hd(@x_train)], @y_train)
    end)

    assert_raise(fn ->
      Tagger.fit(%{}, @x_train, [hd(@y_train)])
    end)

    assert_raise(fn ->
      Tagger.fit(%{}, @x_train, @y_train, algorithm: :invalid)
    end)

    algorithms = [:lbfgs, :l2sgd, :ap, :pa, :arow]
    linesearches = [:more_thuente, :backtracking, :strong_backtracking]

    check all algorithm <- Gen.one_of(algorithms),
              min_freq <- Gen.float(min: 1.0e-5, max: 1),
              all_states? <- Gen.boolean(),
              all_transitions? <- Gen.boolean(),
              c1 <- Gen.float(min: 1.0e-5, max: 1),
              c2 <- Gen.float(min: 1.0e-5, max: 1),
              max_iter <- Gen.positive_integer(),
              num_memories <- Gen.positive_integer(),
              epsilon <- Gen.float(min: 1.0e-5, max: 0.5),
              period <- Gen.positive_integer(),
              delta <- Gen.float(min: 0, max: 1),
              linesearch <- Gen.one_of(linesearches),
              max_linesearch <- Gen.positive_integer(),
              calibration_eta <- Gen.float(min: 1.0e-5, max: 0.5),
              calibration_rate <- Gen.float(min: 0, max: 1),
              calibration_samples <- Gen.positive_integer(),
              calibration_candidates <- Gen.positive_integer(),
              calibration_max_trials <- Gen.positive_integer(),
              pa_type <- Gen.integer(0..2),
              c <- Gen.float(min: 1.0e-5),
              error_sensitive? <- Gen.boolean(),
              averaging? <- Gen.boolean(),
              variance <- Gen.float(min: 0, max: 1),
              gamma <- Gen.float(min: 1.0e-5),
              verbose <- Gen.boolean() do
      options = [
        algorithm: algorithm,
        min_freq: min_freq,
        all_possible_states?: all_states?,
        all_possible_transitions?: all_transitions?,
        c1: c1,
        c2: c2,
        max_iterations: max_iter,
        num_memories: num_memories,
        epsilon: epsilon,
        period: period,
        delta: delta,
        linesearch: linesearch,
        max_linesearch: max_linesearch,
        calibration_eta: calibration_eta,
        calibration_rate: calibration_rate,
        calibration_samples: calibration_samples,
        calibration_candidates: calibration_candidates,
        calibration_max_trials: calibration_max_trials,
        pa_type: pa_type,
        c: c,
        error_sensitive?: error_sensitive?,
        averaging?: averaging?,
        variance: variance,
        gamma: gamma,
        verbose: verbose
      ]

      model = Tagger.fit(%{}, @x_train, @y_train, options)

      y = Tagger.predict_sequence(model, %{}, @x_train)

      for {x, {y_pred, y_prob}} <- Enum.zip(@x_train, y) do
        assert length(y_pred) == length(x)
        assert y_prob >= 0 and y_prob <= 1
      end

      params = Tagger.export(model)
      assert params === Tagger.export(Tagger.compile(params))
    end
  end

  test "featurizer" do
    # string featurizer
    x = [
      ["a", "b"],
      ["x"]
    ]

    expect = [
      [%{"a" => 1.0}, %{"b" => 1.0}],
      [%{"x" => 1.0}]
    ]

    assert Tagger.transform(%{}, %{}, x) === expect

    # list featurizer
    x = [
      [["a1", "a2"], ["b1"]],
      [["x0"]]
    ]

    expect = [
      [%{"a1" => 1.0, "a2" => 1.0}, %{"b1" => 1.0}],
      [%{"x0" => 1.0}]
    ]

    assert Tagger.transform(%{}, %{}, x) === expect

    # map featurizer
    x = [
      [%{:a1 => 4, "a2" => "test"}, %{"b1" => 1.0}],
      [%{"x" => 2.0}]
    ]

    expect = [
      [%{"a1" => 4.0, "a2-test" => 1.0}, %{"b1" => 1.0}],
      [%{"x" => 2.0}]
    ]

    assert Tagger.transform(%{}, %{}, x) === expect
  end

  test "predict" do
    model = Tagger.fit(%{}, @x_train, @y_train)

    y = Tagger.predict_sequence(model, %{}, @x_train)

    for {{y_pred, y_prob}, y_true} <- Enum.zip(y, @y_train) do
      assert y_pred === y_true
      assert y_prob >= 0 and y_prob <= 1
    end

    [{y_pred, y_prob}] = Tagger.predict_sequence(model, %{}, [["some", "unseen", "input"]])

    for y <- y_pred do
      assert y in ["o", "b_num", "i_num", "b_fruit", "i_fruit"]
    end

    assert y_prob >= 0 and y_prob <= 1
  end

  test "global parallelism" do
    tasks =
      Task.async_stream(
        1..1000,
        fn _i ->
          model = Tagger.fit(%{}, @x_train, @y_train)

          params = Tagger.export(model)
          Tagger.compile(params)

          y = Tagger.predict_sequence(model, %{}, @x_train)

          for {{y_pred, y_prob}, y_true} <- Enum.zip(y, @y_train) do
            assert y_pred === y_true
            assert y_prob >= 0 and y_prob <= 1
          end
        end,
        ordered: false
      )

    Stream.run(tasks)
  end

  test "shared parallelism" do
    model = Tagger.fit(%{}, @x_train, @y_train)

    tasks =
      Task.async_stream(
        1..1000,
        fn _i ->
          y = Tagger.predict_sequence(model, %{}, @x_train)

          for {{y_pred, y_prob}, y_true} <- Enum.zip(y, @y_train) do
            assert y_pred === y_true
            assert y_prob >= 0 and y_prob <= 1
          end
        end,
        ordered: false
      )

    Stream.run(tasks)
  end

  @tag :stress
  test "fit stress" do
    for _ <- 1..30_000 do
      Tagger.fit(%{}, @x_train, @y_train)
      :erlang.garbage_collect()
    end
  end

  @tag :stress
  test "export/compile stress" do
    model = Tagger.fit(%{}, @x_train, @y_train)

    for _ <- 1..50_000 do
      params = Tagger.export(model)
      Tagger.compile(params)
      :erlang.garbage_collect()
    end
  end

  @tag :stress
  test "predict stress" do
    model = Tagger.fit(%{}, @x_train, @y_train)

    for _ <- 1..3_000_000 do
      Tagger.predict_sequence(model, %{}, @x_train)
      :erlang.garbage_collect()
    end
  end
end
