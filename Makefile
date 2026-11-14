CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -march=native

all: build/bot build/referee

build/bot: bot.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -DLOCAL -o $@ $<

build/referee: tools/referee.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

perft: build/bot build/referee
	UT_PERFT=8 ./build/bot
	./build/referee --perft 8

bench: build/bot
	UT_BENCH=1 ./build/bot

clean:
	rm -rf build out

.PHONY: all perft bench clean
