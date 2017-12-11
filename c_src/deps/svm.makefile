CC     = g++
CFLAGS = -c -Wall -O3 -fpic \
         -Ilibsvm/include -Ilibsvm/lib/cqdb/include
OBJDIR = ../../obj

all: $(OBJDIR)/svm.o

clean: ; $(RM) $(OBJDIR)/*.o

rebuild: clean all

$(OBJDIR)/svm.o: libsvm/svm.cpp

%.o:
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: all clean rebuild
