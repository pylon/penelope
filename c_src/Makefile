ERLANG_PATH ?= $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
CC = gcc
CFLAGS = -flat_namespace -undefined suppress -fpic -shared -I$(ERLANG_PATH)
OUTDIR = ../priv

all:

clean:
	$(RM) -r $(OUTDIR)

rebuild: clean all

%.so:
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean rebuild watch
