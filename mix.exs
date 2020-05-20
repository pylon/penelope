defmodule Penelope.Mixfile do
  use Mix.Project

  def project do
    [
      app: :penelope,
      name: "Penelope",
      version: "0.5.0",
      elixir: "~> 1.7",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_cwd: "c_src",
      make_clean: ["clean"],
      aliases: [clean: ["clean", "clean.nif"]],
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      description: description(),
      deps: deps(),
      package: package(),
      source_url: "https://github.com/pylon/penelope",
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.html": :test,
        "coveralls.post": :test
      ],
      dialyzer: [
        ignore_warnings: ".dialyzerignore",
        plt_add_deps: :transitive
      ],
      docs: [extras: ["README.md"]]
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
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:e2qc, "~> 1.2"},
      {:poison, "~> 3.0 or ~> 4.0", optional: true},
      {:stream_data, "~> 0.3", only: [:test]},
      {:excoveralls, "~> 0.10", only: :test},
      {:elixir_make, "~> 0.4", runtime: false},
      {:credo, "~> 0.10", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 0.5", only: [:dev], runtime: false},
      {:benchee, "~> 0.13", only: :dev, runtime: false},
      {:ex_doc, "~> 0.19", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      files: [
        "mix.exs",
        "README.md",
        "lib",
        "c_src/**/{*.c,*.cpp,*.h,*.hpp,Makefile,*.makefile}",
        "priv/.gitignore"
      ],
      maintainers: ["Brent M. Spell", "Josh Ziegler"],
      licenses: ["Apache 2.0"],
      links: %{
        "GitHub" => "https://github.com/pylon/penelope",
        "Docs" => "http://hexdocs.pm/penelope/"
      }
    ]
  end
end
