#!/bin/bash
pdsh -w lwacalim[01-08] 'source /home/pipeline/ovro_data_recorder/services/restart_dr_services.sh'
