SHELL:=/usr/bin/bash

default: help

##############################################################################
#
# Machine Learning
#
##############################################################################

INPUT=./data/local/merged_Sliderule/*.csv
MODEL=./models/model.json
EPOCHS=100

.PHONY: score # Compute score across all predictions
score:
	@python apps/score.py \
		--verbose \
		"$(INPUT)"

.PHONY: corr # Compute correlations between predictions
corr:
	@python apps/corr.py \
		--verbose \
		"$(INPUT)"
	@eog all_corr.png &
	@eog surface_corr.png &
	@eog bathy_corr.png &

.PHONY: train # Train a model
train:
	@./apps/train.py \
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
