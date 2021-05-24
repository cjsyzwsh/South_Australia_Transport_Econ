.PHONY: requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################



#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: 
	$(PYTHON_INTERPRETER) pip3 install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) pip3 install -r requirements.txt

## Create intermediate data 
create_intermediate_data:
	$(PYTHON_INTERPRETER) src/d02_intermediate/preprocess_0_shapefiles.py


