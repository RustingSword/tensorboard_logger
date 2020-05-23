PROTOC = protoc
INCLUDES = -Iinclude
LDFLAGS =  -lprotobuf

CC = g++ -std=c++11

PROTOS = $(wildcard proto/*.proto)
SRCS = $(patsubst proto/%.proto,src/%.pb.cc,$(PROTOS))
SRCS += src/tensorboard_logger.cc src/crc.cc
OBJS = $(patsubst src/%.cc,src/%.o,$(SRCS))

LIB = libtensorboard_logger.a

.PHONY: all proto obj test clean distclean lib

all: proto obj lib test
obj: $(OBJS)

proto: $(PROTOS)
	$(PROTOC) -Iproto $(PROTOS) --cpp_out=proto
	mv proto/*.cc src
	mv proto/*.h include

$(OBJS): %.o: %.cc proto
	$(CC) $(INCLUDES) -c $< -o $@

lib: proto obj
	ar rcs $(LIB) $(OBJS)

test: tests/test_tensorboard_logger.cc lib
	$(CC) $(INCLUDES) $< $(LIB) -o $@ $(LDFLAGS)

clean:
	rm -f src/*.o $(LIB) test

distclean: clean
	rm -f include/*.pb.h src/*.pb.cc
