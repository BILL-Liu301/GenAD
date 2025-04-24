#!/bin/bash

mode=with_description_ca_bev
python tools/analysis_tools/visualization.py --result_path test/$mode/pts_bbox/results_nusc.pkl --save_path visualization/$mode
mode=with_description_ca_map
python tools/analysis_tools/visualization.py --result_path test/$mode/pts_bbox/results_nusc.pkl --save_path visualization/$mode
mode=with_description_ca_motion
python tools/analysis_tools/visualization.py --result_path test/$mode/pts_bbox/results_nusc.pkl --save_path visualization/$mode
mode=with_description_ca_map_motion
python tools/analysis_tools/visualization.py --result_path test/$mode/pts_bbox/results_nusc.pkl --save_path visualization/$mode
mode=with_description_ca_bev_map_motion
python tools/analysis_tools/visualization.py --result_path test/$mode/pts_bbox/results_nusc.pkl --save_path visualization/$mode
mode=without_description
python tools/analysis_tools/visualization.py --result_path test/$mode/pts_bbox/results_nusc.pkl --save_path visualization/$mode
