PY ?= $(HOME)/venvs/nikasha/bin/python
ENV := HF_HUB_OFFLINE=1

.PHONY: seta gauge1 timing baseline calibrate score readme selftest canary ku2 all

seta:
	$(ENV) $(PY) -m nikasha.data_seta

timing:
	$(ENV) $(PY) -m nikasha.gauge_logit --timing 50

gauge1:
	$(ENV) $(PY) -m nikasha.gauge_logit --run

baseline:
	$(ENV) $(PY) -m nikasha.gauge_const

calibrate:
	$(ENV) $(PY) -m nikasha.calibrate

score:
	$(ENV) $(PY) -m nikasha.score

canary:
	$(ENV) $(PY) -m nikasha.canary

ku2:
	$(ENV) $(PY) -m nikasha.ku2_hidden

readme:
	$(ENV) $(PY) -m nikasha.readme

selftest:
	$(ENV) $(PY) -m nikasha.selftest

all: seta gauge1 baseline calibrate score canary ku2 readme selftest
