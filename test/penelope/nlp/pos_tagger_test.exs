defmodule Penelope.NLP.POSTaggerTest do
  use ExUnit.Case

  alias Penelope.NLP.POSTagger

  @x_train [
    String.split("Bill is on a boat"),
    String.split("I 'm just a bill")
  ]

  @y_train [
    String.split("NNP VBZ IN DT NN"),
    String.split("PRP VBP RB DT NN")
  ]

  test "fit/export/compile" do
    model = POSTagger.fit(%{}, @x_train, @y_train)
    params = POSTagger.export(model)
    assert params === POSTagger.export(POSTagger.compile(params))
  end

  test "tag" do
    model = POSTagger.fit(%{}, @x_train, @y_train)
    unseen = String.split("Bill is a bill")
    tagged = POSTagger.tag(model, %{}, unseen)
    assert tagged === Enum.zip(unseen, String.split("NNP VBZ DT NN"))
  end
end
