SHELL:=/usr/bin/bash

default: help

##############################################################################
#
# Machine Learning
#
##############################################################################

#INPUT=./ATL24_20190604044922_10220307_006_02.csv
INPUT=./data/local/merged_Sliderule/*.csv
MODEL=./models/model.json
EPOCHS=100

.PHONY: score_all # Compute scores across all predictions
score_all:
	@python apps/score_all.py --verbose "$(INPUT)"

.PHONY: train # Train a model
train:
	@./apps/track_stacker.py \
		--verbose \
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
