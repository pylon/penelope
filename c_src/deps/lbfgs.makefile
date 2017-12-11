CC     = gcc
CFLAGS = -c -Wall -O3 -fpic \
         -Iliblbfgs/include
OBJDIR = ../../obj

all: $(OBJDIR)/lbfgs.o

clean: ; $(RM) $(OBJDIR)/*.o

rebuild: clean all

$(OBJDIR)/lbfgs.o: liblbfgs/lib/lbfgs.c

%.o:
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean rebuild
