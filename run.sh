#! /usr/bin/env bash

jetson-containers run -v ~/Downloads:/model-cache cascade-pipeline:r36.3.0 $@
