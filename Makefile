SHELL:=/usr/bin/bash

default: help

##############################################################################
#
# Machine Learning
#
##############################################################################

INPUT=./data/local/combined/*.csv
MODEL=./models/model.json
EPOCHS=100

.PHONY: train # Train a model
train:
	@./apps/track_stacker.py \
		--verbose \
		--model-filename=$(MODEL) \
		"$(INPUT)"

##############################################################################
#
# Get help by running
#
#     $ make help
#
##############################################################################
.PHONY: help # Generate list of targets with descriptions
help:
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t25
