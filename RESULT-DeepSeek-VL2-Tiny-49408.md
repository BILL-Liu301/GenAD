[BEV, Map, Motion]

# Result of DesEAD

## Motion Prediction

### CAR

|                                         |       EPA(👆)       |      ADE(👇)      |      FDE(👇)      |       MR(👇)       |
| --------------------------------------- | :-----------------: | :----------------: | :----------------: | :-----------------: |
| Without Description                     | 0.07068366164542295 | 1.5051437616348267 | 1.9550225734710693 | 0.15217391304347827 |
| With Description, CA [Map, Motion]      | 0.06257242178447277 | 1.461746096611023 | 2.045318365097046 | 0.17518248175182483 |
| With Description, CA [BEV, Map, Motion] | 0.04519119351100811 | 1.6221948862075806 | 2.449563980102539 | 0.3076923076923077 |

### Pedestrian

|                                         |        EPA(👆)        |      ADE(👇)      |      FDE(👇)      |       MR(👇)       |
| --------------------------------------- | :--------------------: | :---------------: | :----------------: | :-----------------: |
| Without Description                     | -0.0030349013657056147 | 1.625779628753662 | 2.593676805496216 | 0.5833333333333334 |
| With Description, CA [Map, Motion]      |  0.028831562974203338  | 1.222456693649292 | 1.9383858442306519 | 0.46153846153846156 |
| With Description, CA [BEV, Map, Motion] |  0.002276176024279211  | 1.765453815460205 | 3.146254062652588 | 0.5517241379310345 |

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

|                                         |       1s(👇)       |       2s(👇)       |      3s(👇)      |
| :-------------------------------------- | :----------------: | :----------------: | :---------------: |
| Without Description                     | 2.217079136574614 | 3.7318169189536055 | 5.356779453115187 |
| With Description, CA [Map, Motion]      | 2.1525086000345754 | 3.584020824967951 | 5.084014963844548 |
| With Description, CA [BEV, Map, Motion] | 1.8513372662490692 | 3.294419193181439 | 4.921352621437847 |

### Collision Rate(👇)

|                                         | 1s(👇) |        2s(👇)        |       3s(👇)       |
| --------------------------------------- | ------ | :------------------: | :-----------------: |
| Without Description                     | 0.0    | 0.014492753623188406 | 0.0386473439309908 |
| With Description, CA [Map, Motion]      | 0.0    | 0.007246376811594203 | 0.03623188470584759 |
| With Description, CA [BEV, Map, Motion] | 0.0    | 0.010869565217391304 | 0.05072463854499485 |

## Evaluate bboxes of pts_box

|                                         | mAP(👆) |  mATE  |  mASE  | mAOE   | mAVE   | mAAE   |  NDS  |
| :-------------------------------------- | :-----: | :----: | :----: | ------ | ------ | ------ | :----: |
| Without Description                     | 0.0097 | 1.0658 | 0.8955 | 1.0042 | 0.9794 | 0.9400 | 0.0234 |
| With Description, CA [Map, Motion]      | 0.0102 | 0.9889 | 0.8589 | 1.0861 | 0.9331 | 0.8821 | 0.0388 |
| With Description, CA [BEV, Map, Motion] | 0.0070 | 1.0374 | 0.9036 | 1.0660 | 0.9981 | 0.9528 | 0.0180 |

## Map Segmentation (我不确定这个数据有没有问题)

### mAP(👆)

|                                         | 0.5(👆) |        1.0(👆)        |       1.5(👆)       |
| --------------------------------------- | ------- | :-------------------: | :------------------: |
| Without Description                     | 0.000   | 0.005492950323969126 | 0.019970213994383812 |
| With Description, CA [Map, Motion]      | 0.000   | 0.0010877320310100913 | 0.008329629898071289 |
| With Description, CA [BEV, Map, Motion] | 0.000   | 0.0018472153460606933 | 0.014297413639724255 |
