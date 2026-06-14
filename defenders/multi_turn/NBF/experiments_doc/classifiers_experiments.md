# experiment1

full features no dimensionality reduction<br>
features= [xt,zt]<br>
SVC(kernel='rbf', C=1.0, gamma='scale')

```
Accuracy: 0.5798826777087647
              precision    recall  f1-score   support

           1       0.68      0.69      0.69      1467
           2       0.50      0.30      0.38       927
           3       0.54      0.75      0.63      2173
           4       0.57      0.38      0.45       682
           5       0.64      0.34      0.45       547

    accuracy                           0.58      5796
   macro avg       0.59      0.49      0.52      5796
weighted avg       0.58      0.58      0.56      5796
```

hyperparameter results
```
params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
```
results<br>
Best params: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}

===========================================================
# experiment2
full features no dimensionality reduction<br>
features= [xt,ut]<br>
SVC(kernel='rbf', C=1.0, gamma='scale')

```
Accuracy: 0.5707384403036577
              precision    recall  f1-score   support

           1       0.68      0.67      0.68       718
           2       0.51      0.29      0.37       468
           3       0.53      0.76      0.62      1084
           4       0.55      0.33      0.41       359
           5       0.60      0.36      0.45       269

    accuracy                           0.57      2898
   macro avg       0.57      0.48      0.51      2898
weighted avg       0.57      0.57      0.55      2898
```

===========================================================
# experiment 3
```

Number of components: 212
Total explained variance: 0.95000315
features: PCA[xt,zt]
Accuracy: 0.5778122843340234
              precision    recall  f1-score   support

           1       0.68      0.69      0.68      1467
           2       0.50      0.30      0.37       927
           3       0.54      0.75      0.63      2173
           4       0.56      0.37      0.45       682
           5       0.63      0.35      0.45       547

    accuracy                           0.58      5796
   macro avg       0.58      0.49      0.52      5796
weighted avg       0.58      0.58      0.56      5796

```

======================================================
# experiment 4

```
Number of components: 352
Total explained variance: 0.9501066
features:PCA[xt,ut]
Accuracy: 0.5871290545203589
              precision    recall  f1-score   support

           1       0.69      0.68      0.69      1467
           2       0.54      0.33      0.41       927
           3       0.55      0.76      0.64      2173
           4       0.56      0.37      0.45       682
           5       0.62      0.37      0.46       547

    accuracy                           0.59      5796
   macro avg       0.59      0.50      0.53      5796
weighted avg       0.59      0.59      0.57      5796
```