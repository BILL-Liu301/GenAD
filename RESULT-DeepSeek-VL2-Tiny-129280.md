[BEV, Map, Motion]

# Result of DesEAD

## Motion Prediction

### CAR

|                                         |       EPA(👆)       |      ADE(👇)      |      FDE(👇)      |       MR(👇)       |
| --------------------------------------- | :-----------------: | :----------------: | :----------------: | :-----------------: |
| Without Description                     | 0.07068366164542295 | 1.5051437616348267 | 1.9550225734710693 | 0.15217391304347827 |
| With Description, CA [Map, Motion]      | 0.07937427578215528 | 1.7316970825195312 | 2.317997932434082 | 0.28104575163398693 |
| With Description, CA [BEV, Map, Motion] | 0.03997682502896872 | 1.6221948862075806 | 2.7260942459106445 | 0.2537313432835821 |

### Pedestrian

|                                         |        EPA(👆)        |      ADE(👇)      |      FDE(👇)      |       MR(👇)       |
| --------------------------------------- | :--------------------: | :----------------: | :----------------: | :----------------: |
| Without Description                     | -0.0030349013657056147 | 1.625779628753662 | 2.593676805496216 | 0.5833333333333334 |
| With Description, CA [Map, Motion]      | -0.006069802731411229 | 1.869293212890625 | 3.7322018146514893 | 0.8461538461538461 |
| With Description, CA [BEV, Map, Motion] | -0.0015174506828528073 | 2.1131911277770996 | 3.3464572429656982 |        0.8        |

## Plan

### 不确定的指标

GenAD的代码中输出了很多我不确定什么意思的指标

* [X] plan_L2_xs
* [ ] plan_obj_col_xs
* [X] plan_obj_box_col_xs
* [ ] plan_L2_stp3_xs
* [ ] plan_obj_col_stp3_xs
* [ ] plan_obj_box_col_stp3_xs

```
metric_dict['plan_L2_{}s'.format(i+1)] =traj_L2
metric_dict['plan_L2_stp3_{}s'.format(i+1)] =traj_L2_stp3
metric_dict['plan_obj_col_{}s'.format(i+1)] =obj_coll.mean().item()
metric_dict['plan_obj_col_stp3_{}s'.format(i+1)] =obj_coll[-1].item()
metric_dict['plan_obj_box_col_{}s'.format(i+1)] =obj_box_coll.mean().item()
metric_dict['plan_obj_box_col_stp3_{}s'.format(i+1)] =obj_box_coll[-1].item()
```

### L2(👇)

|                                         |       1s(👇)       |       2s(👇)       |       3s(👇)       |
| :-------------------------------------- | :----------------: | :----------------: | :----------------: |
| Without Description                     | 2.217079136574614 | 3.7318169189536055 | 5.356779453115187 |
| With Description, CA [Map, Motion]      | 1.9064157447521237 | 3.1996362747057625 | 4.5997704025627915 |
| With Description, CA [BEV, Map, Motion] | 2.0800990492537403 | 3.441465832616972 | 4.828680089418439 |

### Collision Rate(👇)

|                                         | 1s(👇) |        2s(👇)        |        3s(👇)        |
| --------------------------------------- | ------ | :------------------: | :------------------: |
| Without Description                     | 0.0    | 0.014492753623188406 |  0.0386473439309908  |
| With Description, CA [Map, Motion]      | 0.0    | 0.007246376811594203 | 0.038647343715031944 |
| With Description, CA [BEV, Map, Motion] | 0.0    | 0.007246376811594203 | 0.03381642591262209 |

## Evaluate bboxes of pts_box

|                                         | mAP(👆) |  mATE  |  mASE  | mAOE   | mAVE   | mAAE   |  NDS  |
| :-------------------------------------- | :-----: | :----: | :----: | ------ | ------ | ------ | :----: |
| Without Description                     | 0.0097 | 1.0658 | 0.8955 | 1.0042 | 0.9794 | 0.9400 | 0.0234 |
| With Description, CA [Map, Motion]      | 0.0094 | 0.9949 | 0.9223 | 0.9964 | 0.9659 | 0.9272 | 0.0240 |
| With Description, CA [BEV, Map, Motion] | 0.0066 | 1.0575 | 0.9013 | 1.0498 | 0.9924 | 0.9500 | 0.0189 |

## Map Segmentation (我不确定这个数据有没有问题)

### mAP(👆)

|                                         | 0.5(👆) |        1.0(👆)        |       1.5(👆)       |
| --------------------------------------- | ------- | :-------------------: | :------------------: |
| Without Description                     | 0.000   | 0.005492950323969126 | 0.019970213994383812 |
| With Description, CA [Map, Motion]      | 0.000   | 0.0039447457529604435 | 0.022952275350689888 |
| With Description, CA [BEV, Map, Motion] | 0.000   | 0.0006479075527749956 | 0.007397944573312998 |
