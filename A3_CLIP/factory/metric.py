import sklearn
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score,matthews_corrcoef

# gt_np = [0,1,1,0]
# pred_np = [0.2,0.7,0.8,0.3]
# fps,tps,thresholds = sklearn.metrics._ranking._binary_clf_curve(gt_np, pred_np)
# precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
# print(fps,tps,thresholds) # tps An increasing count of true positives
# print(precision[:-1], recall[:-1], thresholds)
# tns = tps/recall[:-1] - tps

# print(tns + fps + tps)
# fns = 
# [0. 1. 2.] [2. 2. 2.] [0.8 0.3 0.2]
# precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
# numerator = 2 * recall * precision
# denom = recall + precision
# f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
# max_f1 = np.max(f1_scores)
# max_f1_thresh = thresholds[np.argmax(f1_scores)]
# print('The max_f1_thresh is', max_f1_thresh)
# print('The average f1_score is', max_f1)
# print('The average acc_score is', accuracy_score(gt_np, pred_np>max_f1_thresh))   
