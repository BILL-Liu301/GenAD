#!/bin/bash

mode=with_description_ca_bev
python tools/test.py outputs/$mode/GenAD_config.py outputs/$mode/latest.pth --launcher none --eval bbox --tmpdir testoutputs
mode=with_description_ca_map
python tools/test.py outputs/$mode/GenAD_config.py outputs/$mode/latest.pth --launcher none --eval bbox --tmpdir testoutputs
mode=with_description_ca_motion
python tools/test.py outputs/$mode/GenAD_config.py outputs/$mode/latest.pth --launcher none --eval bbox --tmpdir testoutputs
mode=with_description_ca_map_motion
python tools/test.py outputs/$mode/GenAD_config.py outputs/$mode/latest.pth --launcher none --eval bbox --tmpdir testoutputs
mode=with_description_ca_bev_map_motion
python tools/test.py outputs/$mode/GenAD_config.py outputs/$mode/latest.pth --launcher none --eval bbox --tmpdir testoutputs
mode=without_description
python tools/test.py outputs/$mode/GenAD_config.py outputs/$mode/latest.pth --launcher none --eval bbox --tmpdir testoutputs
