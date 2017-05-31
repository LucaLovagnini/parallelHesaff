appname := hesaff

PARALLELOPENCV=$(HOME)/ParallelOpenCV/install_icpc
INCLUDE=$(PARALLELOPENCV)/include
OPENCVLIB=$(PARALLELOPENCV)/lib

INTEL_INSPECTOR=-g -O0 -qopenmp-link dynamic -qopenmp
INTEL_OPT=-O3 -ipo -simd -xCORE-AVX2 -parallel -qopenmp -fargument-noalias -ansi-alias -no-prec-div -fp-model fast=2 -fma -align -finline-functions
#better not include -ipo for advisor and -qopt-report
#-ipo -fargument-noalias -ansi-alias -no-prec-div -fp-model fast=2 -fma -align -finline-functions

INTEL_PROFILE=-g -qopt-report=5 -Bdynamic -shared-intel -debug inline-debug-info -qopenmp-link dynamic -parallel-source-info=2 -ldl

CXX := icpc
CXXFLAGS := -DRATIOTHRESHOLD=0.23 -I$(INCLUDE) $(INTEL_PROFILE) $(INTEL_OPT) -std=c++11

LDFLAGS= -L$(OPENCVLIB) $(INTEL_PROFILE) $(INTEL_OPT)
LDLIBS= -ltbb -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

srcfiles := $(shell find . -name "*.cpp")
headerfiles := $(shell find . \( -name "*.hpp" -o -name "*.h" \))

objects := $(patsubst %.cpp,%.o,$(notdir $(srcfiles)))

VPATH := $(sort $(dir $(srcfiles)))

all: $(srcfiles) $(appname)

$(appname): $(objects)
	$(CXX) -o $(appname) $(objects) $(LDLIBS) $(LDFLAGS)

depend: .depend

.depend: $(srcfiles) $(headerfiles)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) $(CXXDISALBE) -MM $^>>./.depend;

clean:
	rm -f $(appname) $(objects)

dist-clean: clean
	rm -f *~ .depend

include .depend

