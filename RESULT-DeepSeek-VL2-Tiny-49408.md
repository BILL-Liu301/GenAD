[BEV, Map, Motion]

# Result of DesEAD

## Motion Prediction

### CAR

|                                         |      EPA(👆)      |      ADE(👇)      |      FDE(👇)      |       MR(👇)       |
| --------------------------------------- | :----------------: | :----------------: | :----------------: | :-----------------: |
| Without Description                     | 0.5542498709754258 | 0.7996403574943542 | 1.0832337141036987 | 0.11349408764201252 |
| With Description, CA [BEV, Map, Motion] | 0.5277502084243122 | 0.934629499912262 | 1.3197948932647705 | 0.14209930261667758 |

### Pedestrian

|                                         |       EPA(👆)       |      ADE(👇)      |      FDE(👇)      |       MR(👇)       |
| --------------------------------------- | :-----------------: | :----------------: | :----------------: | :-----------------: |
| Without Description                     | 0.3139599813866915 | 0.8164960741996765 | 1.0935914516448975 | 0.1387270282384581 |
| With Description, CA [BEV, Map, Motion] | 0.25746859004187994 | 0.8941882252693176 | 1.2481939792633057 | 0.17321997874601489 |

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
| :-------------------------------------- | :-----------------: | :----------------: | :----------------: |
| Without Description                     | 0.2960830323897006 | 0.5287353468178684 | 0.8357996264593465 |
| With Description, CA [BEV, Map, Motion] | 0.30053911283543683 | 0.5376617590452569 | 0.8466211300260995 |

### Collision Rate(👇)

|                                         | 1s(👇)                |        2s(👇)        |        3s(👇)        |
| --------------------------------------- | --------------------- | :-------------------: | :-------------------: |
| Without Description                     | 0.0019535065442469234 | 0.0029302598163703846 | 0.0038744547208041553 |
| With Description, CA [BEV, Map, Motion] | 0.0022465325258839617 | 0.0032232857980074234 | 0.004330272881470986 |

## Evaluate bboxes of pts_box

|                                         | mAP(👆) |  mATE  |  mASE  | mAOE   | mAVE   | mAAE   |  NDS  |
| :-------------------------------------- | :-----: | :----: | :----: | ------ | ------ | ------ | :----: |
| Without Description                     | 0.2053 | 0.7557 | 0.5213 | 0.8225 | 0.6690 | 0.5184 | 0.2740 |
| With Description, CA [BEV, Map, Motion] | 0.1912 | 0.7639 | 0.5213 | 0.8468 | 0.9644 | 0.5234 | 0.2336 |
