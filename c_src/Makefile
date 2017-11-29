ERLANG_PATH ?= $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version)])])' -s init stop -noshell)
CC           = g++
CFLAGS       = -Wall -O3 -fpic -Wl,-undefined,dynamic_lookup -shared \
               -I$(ERLANG_PATH)/include -L$(ERLANG_PATH)/lib
LIBS         = -lerts
OBJDIR       = ../obj
OUTDIR       = ../priv

ifeq ($(shell test -f /usr/lib/libopenblas.so && echo -n yes), yes)
   LIBS += -lopenblas
else
   LIBS += -lblas
endif

all: $(OUTDIR)/penelope.so

clean:
	$(RM) $(OBJDIR)/*.o
	$(RM) $(OUTDIR)/penelope.so

rebuild: clean all

$(OUTDIR)/penelope.so: init.cpp blas.cpp svm.cpp

%.so:
	mkdir -p $(dir $@)
	cd deps && $(MAKE) -f svm.makefile
	cd deps && $(MAKE) -f lbfgs.makefile
	cd deps && $(MAKE) -f crfsuite.makefile
	$(CC) $(CFLAGS) -o $@ $^ $(OBJDIR)/*.o $(LIBS)

.PHONY: all clean rebuild
