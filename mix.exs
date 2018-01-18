defmodule Penelope.Mixfile do
  use Mix.Project

  def project do
    [
      app:               :penelope,
      name:              "Penelope",
      version:           "0.3.0",
      elixir:            "~> 1.6",
      compilers:         ["nif" | Mix.compilers],
      aliases:           [clean: ["clean", "clean.nif"]],
      elixirc_paths:     elixirc_paths(Mix.env),
      start_permanent:   Mix.env == :prod,
      description:       description(),
      deps:              deps(),
      package:           package(),
      test_coverage:     [tool: ExCoveralls],
      preferred_cli_env: [coveralls: :test, "coveralls.html": :test],
      dialyzer:          [ignore_warnings: ".dialyzerignore",
                          plt_add_deps:    :transitive],
      docs:              [extras: ["README.md"]]
    ]
  end

  defp description do
    """
    Natural Language Processing (NLP) and Machine Learning (ML) library for
    Elixir.
    """
  end

  def application do
    [
      mod: {Penelope.Application, []},
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_),     do: ["lib"]

  defp deps do
    [
      {:e2qc, "~> 1.2"},
      {:stream_data, "~> 0.3", only: [:test]},
      {:excoveralls, "~> 0.8", only: :test},
      {:credo, "~> 0.8", only: [:dev, :test], runtime: false},
      {:dogma, "~> 0.1", only: [:dev], runtime: false},
      {:dialyxir, "~> 0.5", only: [:dev], runtime: false},
      {:benchee, "~> 0.9", only: :dev, runtime: false},
      {:ex_doc, "~> 0.18", only: :dev, runtime: false}
    ]
  end

  defp package do
  [
    files:       ["mix.exs", "README.md", "lib", "c_src", "priv/.gitignore"],
    maintainers: ["Brent M. Spell", "Josh Ziegler"],
    licenses:    ["Apache 2.0"],
    links:       %{"GitHub" => "https://github.com/pylon/penelope",
                   "Docs"   => "http://hexdocs.pm/penelope/"}
   ]
  end
end

defmodule Mix.Tasks.Compile.Nif do
  def run(_args) do
    {result, _errcode} = System.cmd("make", ["-C", "c_src"])
    IO.binwrite(result)
  end
end

defmodule Mix.Tasks.Clean.Nif do
  def run(_args) do
    {result, _errcode} = System.cmd("make", ["-C", "c_src", "clean"])
    IO.binwrite(result)
  end
end
