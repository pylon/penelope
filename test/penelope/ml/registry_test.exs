defmodule Penelope.ML.RegistryTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Registry

  import Penelope.TestUtility

  defmodule TestModule1 do
  end

  defmodule TestModule2 do
  end

  defmodule TestModule3 do
  end

  test "default entries" do
    assert Registry.lookup("svm_classifier") === Penelope.ML.SVM.Classifier
    assert Registry.lookup(:svm_classifier) === Penelope.ML.SVM.Classifier
    assert Registry.invert(Penelope.ML.SVM.Classifier) === "svm_classifier"
  end

  test "entry registration" do
    assert Registry.register("mymodule1", TestModule1) === :ok
    assert Registry.register(:mymodule2, TestModule2) === :ok

    assert Registry.lookup("mymodule1") === TestModule1
    assert Registry.lookup(:mymodule1) === TestModule1

    assert Registry.lookup("mymodule2") === TestModule2
    assert Registry.lookup(:mymodule2) === TestModule2
  end

  test "module fallback" do
    assert_raise(fn ->
      Registry.lookup(Invalid.Module)
    end)

    assert Registry.lookup("Elixir.Penelope.ML.RegistryTest.TestModule3") ===
             TestModule3

    assert Registry.lookup(TestModule3) === TestModule3

    assert Registry.invert(TestModule3) ===
             "Elixir.Penelope.ML.RegistryTest.TestModule3"
  end
end
