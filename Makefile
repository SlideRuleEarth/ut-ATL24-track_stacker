SHELL:=/usr/bin/bash

default: help

##############################################################################
#
# Machine Learning
#
##############################################################################

INPUT=./data/remote/latest/*.csv
OUTPUT_DIR=./predictions
MODEL=./models/model.json
EPOCHS=100

.PHONY: check_hashes # Check git local and remote repo hashs
check_hashes:
	@bash ./scripts/check_git_hashes.bash

.PHONY: train # Train a model
train: check_hashes
	@./apps/train.py \
		--verbose \
		--epochs=$(EPOCHS) \
		--model-filename=$(MODEL) \
		"$(INPUT)"

.PHONY: classify # Generate predictions
classify: check_hashes
	@mkdir -p $(OUTPUT_DIR)
	@ls -1 $(INPUT) \
		| parallel --verbose --lb --jobs=16 --halt now,fail=1 \
		"python apps/classify.py --verbose --model-filename=$(MODEL) --output-filename=$(OUTPUT_DIR)/{/.}_classified.csv {}"

.PHONY: score # Score predictions
score:
	make --no-print-directory score_all > scores.all.txt
	make --no-print-directory score_binary > scores.binary.txt

score_all:
	@python apps/score.py --verbose --all "$(OUTPUT_DIR)/*.csv"

score_binary:
	@python apps/score.py --verbose "$(OUTPUT_DIR)/*.csv"

.PHONY: cross_validate # Cross validate track stacker
cross_validate:
	@python ./apps/generate_cross_val_commands.py \
		--verbose \
		--splits=5 \
		"$(INPUT)" > ./cross_validate.bash
	@bash ./cross_validate.bash
	@rm ./cross_validate.bash

##############################################################################
#
# View results
#
##############################################################################

.PHONY: view # View predictions
view:
	@parallel --lb --jobs=100 \
		"streamlit run ../ATL24_viewer/view_ensemble.py -- --verbose {}" \
		::: $$(find ./predictions/*.csv | head)

##############################################################################
#
# Show result plots
#
##############################################################################

.PHONY: plot_corr # Plot correlations between predictions
plot_corr:
	@python apps/plot_corr.py \
		--verbose \
		"$(INPUT)"
	@eog all_corr.png &
	@eog surface_corr.png &
	@eog bathy_corr.png &

.PHONY: plot_multi_class # Plot performance
plot_multi_class:
	@python ./apps/plot_multi_class.py scores.all.txt
	@python ./apps/plot_multi_class.py cross_val.all.0.txt
	@python ./apps/plot_multi_class.py cross_val.all.1.txt
	@python ./apps/plot_multi_class.py cross_val.all.2.txt
	@python ./apps/plot_multi_class.py cross_val.all.3.txt
	@python ./apps/plot_multi_class.py cross_val.all.4.txt

.PHONY: plot_binary # Plot performance
plot_binary:
	@python ./apps/plot_binary.py scores.binary.txt
	@python ./apps/plot_binary.py cross_val.binary.0.txt
	@python ./apps/plot_binary.py cross_val.binary.1.txt
	@python ./apps/plot_binary.py cross_val.binary.2.txt
	@python ./apps/plot_binary.py cross_val.binary.3.txt
	@python ./apps/plot_binary.py cross_val.binary.4.txt

.PHONY: plot_f1 # Plot performance
plot_f1:
	@python ./apps/plot_f1.py scores.binary.txt
	@python ./apps/plot_f1.py cross_val.binary.0.txt
	@python ./apps/plot_f1.py cross_val.binary.1.txt
	@python ./apps/plot_f1.py cross_val.binary.2.txt
	@python ./apps/plot_f1.py cross_val.binary.3.txt
	@python ./apps/plot_f1.py cross_val.binary.4.txt

.PHONY: plot # Plot performance
plot:
	@python ./apps/plot_multi_class.py scores.all.txt
	@python ./apps/plot_surface_bathy.py scores.binary.txt
	@python ./apps/plot_surface_bathy.py cross_val.binary.?.txt

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
