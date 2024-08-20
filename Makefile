SHELL:=/usr/bin/bash

default: help

##############################################################################
#
# Machine Learning
#
##############################################################################

INPUT=./data/local/merged_Sliderule_v1/*.csv
OUTPUT=./predictions/*.csv
MODEL=./models/model.json
EPOCHS=100

.PHONY: score_algorithms # Compute score for individual algorithms
score_algorithms:
	@python apps/score_algorithms.py \
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

.PHONY: classify # Generate predictions
classify:
	@mkdir -p predictions
	@ls -1 $(INPUT) \
		| head \
		| parallel --verbose --lb --jobs=16 --halt now,fail=1 \
		"python apps/classify.py --verbose --model-filename=$(MODEL) --output-filename=predictions/{/.}_classified.csv {}"

.PHONY: score # Score predictions
score:
	@python apps/score.py "$(OUTPUT)"

.PHONY: cross_validate # Cross validate track stacker
cross_validate:
	@python ./scripts/generate_cross_val_commands.py \
		--verbose \
		--splits=5 \
		"$(INPUT)" > ./cross_validate.bash
	@bash ./cross_validate.bash
	@rm ./cross_validate.bash

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
