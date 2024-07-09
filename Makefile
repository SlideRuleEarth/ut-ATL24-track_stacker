SHELL:=/usr/bin/bash

default: help

##############################################################################
#
# Machine Learning
#
##############################################################################

INPUT=./data/local/merged_Sliderule/*_1.csv
MODEL=./models/model.json
EPOCHS=100

.PHONY: score # Compute score across all predictions
score:
	@python apps/score.py \
		"$(INPUT)"

.PHONY: train # Train a model
train:
	@./apps/train.py \
		--epochs=$(EPOCHS) \
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
