from ubuntu:latest

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update && \
    apt-get -y install wget locales build-essential && \
    locale-gen en_US.UTF-8

RUN wget https://packages.erlang-solutions.com/erlang-solutions_1.0_all.deb && \
    dpkg -i erlang-solutions_1.0_all.deb && \
    apt-get update && \
    apt-get -y install esl-erlang elixir

RUN mix local.hex --force && \
    mix local.rebar --force

RUN apt-get -y install libblas-dev

ADD mix.exs mix.lock ./
RUN mix do deps.get, deps.compile

ADD c_src ./c_src
ADD lib ./lib
ADD test ./test
RUN mkdir priv

ENTRYPOINT ["mix"]
