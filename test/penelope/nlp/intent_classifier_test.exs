defmodule Penelope.NLP.IntentClassifierTest do
  use ExUnit.Case, async: true

  alias Penelope.NLP.IntentClassifier

  @pipeline [
    tokenizer: [{Penelope.NLP.IntentClassifierTest.Tokenizer, []}],
    classifier: [{Penelope.NLP.IntentClassifierTest.Classifier, []}],
    recognizer: [{Penelope.NLP.IntentClassifierTest.Recognizer, []}]
  ]

  @x_train ["", "you have four pears", "these one hundred apples"]
  @y_train [
    {"intent_1", []},
    {"intent_2", ["o", "o", "b_num", "b_fruit"]},
    {"intent_3", ["o", "b_num", "i_num", "b_fruit"]}
  ]
  @y_param [
    %{},
    %{"num" => "four", "fruit" => "pears"},
    %{"num" => "one hundred", "fruit" => "apples"}
  ]

  test "fit/export/compile" do
    model = IntentClassifier.fit(%{}, @x_train, @y_train, @pipeline)

    params = IntentClassifier.export(model)

    assert params ===
             IntentClassifier.export(IntentClassifier.compile(params))
  end

  test "predict" do
    model = IntentClassifier.fit(%{}, @x_train, @y_train, @pipeline)

    defaults = %{
      "intent_1" => 0.0,
      "intent_2" => 0.0,
      "intent_3" => 0.0
    }

    unseen = IntentClassifier.predict_intent(model, %{}, "unseen sample")
    assert unseen === {Map.put(defaults, "intent_0", 1.0), %{}}

    for {x, y, p} <- Enum.zip([@x_train, @y_train, @y_param]) do
      {y_intent, _y_tags} = y

      assert IntentClassifier.predict_intent(model, %{}, x) ===
               {Map.put(defaults, y_intent, 1.0), p}
    end
  end

  defmodule Tokenizer do
    def transform(_model, _context, x) do
      Enum.map(x, &do_transform/1)
    end

    defp do_transform(x) when is_binary(x) do
      String.split(x, " ", trim: true)
    end

    defp do_transform(x) when is_list(x) do
      Enum.join(x, " ")
    end
  end

  defmodule Classifier do
    def fit(_context, x, y, _options) do
      %{memo: Map.new(Enum.zip(x, y)), classes: Enum.uniq(y)}
    end

    def predict_probability(model, _context, x) do
      Enum.map(x, fn x ->
        c = Map.get(model.memo, x, "intent_0")
        p = Map.new(model.classes, &{&1, 0.0})
        Map.put(p, c, 1.0)
      end)
    end
  end

  defmodule Recognizer do
    def fit(_context, x, y, _options) do
      %{memo: Map.new(Enum.zip(x, y))}
    end

    def predict_sequence(model, _context, x) do
      Enum.map(x, fn x ->
        default = Enum.map(x, fn _x -> "o" end)
        {Map.get(model.memo, x, default), 1.0}
      end)
    end
  end
end
