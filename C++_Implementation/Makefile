#Set OS
UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
	include Makefile_gnu
else ifeq ($(UNAME), Darwin) 
	include Makefile_osx
endif

#Set Directories
source_dir = src
build_dir = build
binary_dir = bin

makefiles = $(wildcard Makefile*)
src = $(wildcard $(source_dir)/*.cpp)

objs = $(src:%.cpp=$(build_dir)/%.o) 
deps = $(src:%.cpp=$(build_dir)/%.d)


all: main
main: $(binary_dir)/main 

# Compiling the main program
$(binary_dir)/main: $(build_dir)/main.o $(objs)
	@mkdir -p $(@D)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

# Building general object files and dependencies
# cpp files
$(build_dir)/%.o: %.cpp $(makefiles)
	@mkdir -p $(@D)
	$(COMPILER) $(CXXFLAGS) -MMD -c $< -o $@


.PHONY: clean main

clean:
	@$(RM) $(objs) $(deps) 

veryclean: clean
	@$(RM) -r $(build_dir) $(binary_dir)

#Include Dependencies
-include $(deps)