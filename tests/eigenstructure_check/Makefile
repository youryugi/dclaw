# Eigenstructure sanity checks: contact-wave preservation and dry dam-break
# front-speed bound, checked against the closed-form results in Iverson &
# George (2014) eq 2.17-2.26 and the classical shallow-water dam-break
# solution. See setinput.py for details.
CLAWMAKE = $(CLAW)/clawutil/src/Makefile.common
CLAW_PKG = dclaw
EXE = $(CURDIR)/xdclaw
SETRUN_FILE = setrun.py
OUTDIR ?= _output

FC = gfortran
FFLAGS ?= -O -gno-strict-dwarf -fbounds-check -fopenmp \
          -std=legacy -ffpe-trap='invalid,overflow,zero'

DIGLIB = $(CLAW)/dclaw/src/2d/dig
AMRLIB = $(CLAW)/amrclaw/src/2d
GEOLIB = $(CLAW)/geoclaw/src/2d/shallow
include $(DIGLIB)/Makefile.dclaw

EXCLUDE_MODULES =
EXCLUDE_SOURCES =
MODULES =
SOURCES =

include $(CLAWMAKE)

input:
	$(CLAW_PYTHON) setinput.py
