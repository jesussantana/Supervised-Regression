# Analize Models
# ==============================================================================
# Analize Model Score & print Confuxion Matrix
# ==============================================================================


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, euclidean_distances
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, plot_confusion_matrix
from sklearn.metrics import precision_score, f1_score, classification_report, balanced_accuracy_score

# ==============================================================================


def models(model, name, X_train, X_test, y_train, y_test):
    
    model.fit(X_train,y_train)    
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test,y_pred,normalize='true')

    scores = {'model': str(name),
            'Accuracy':accuracy_score(y_test,y_pred),
            'B-Accuracy':balanced_accuracy_score(y_test,y_pred),
            'Roc_AUC':roc_auc_score(y_test,y_pred),
            'F1 Score':f1_score(y_test,y_pred),
            'Precision':precision_score(y_test,y_pred),
            'Recall':recall_score(y_test,y_pred)}

    print(model)
    print('')
    print(f'Accuracy : {accuracy_score(y_test,y_pred)}')
    print(f'B-Accuracy :{balanced_accuracy_score(y_test,y_pred)}')
    print(f'ROC - AUC : {roc_auc_score(y_test,y_pred)}')
    print(f'F1 Score : {f1_score(y_test,y_pred)}')
    print(f'Precision : {precision_score(y_test,y_pred)}')
    print(f'Recall : {recall_score(y_test,y_pred)}')
    print('')
    print(conf_matrix)
    print('')

    plot_confusion_matrix(model, X_test, y_test, normalize='true')

    return scores


def AutoML(model, name, X_test, y_test):
    
    
    conf_matrix = confusion_matrix(y_test,model.predict(X_test),normalize='true')

    scores = {'model': str(name),
            'Accuracy':accuracy_score(y_test, model.predict(X_test)),
            'B-Accuracy':balanced_accuracy_score(y_test, model.predict(X_test)),
            'Roc_AUC':roc_auc_score(y_test, model.predict(X_test)),
            'F1 Score':f1_score(y_test, model.predict(X_test)),
            'Precision':precision_score(y_test, model.predict(X_test)),
            'Recall':recall_score(y_test,model.predict(X_test))}

    print(model)
    print('')
    print(f'Accuracy : {accuracy_score(y_test, model.predict(X_test))}')
    print(f'B-Accuracy :{balanced_accuracy_score(y_test,model.predict(X_test))}')
    print(f'ROC - AUC : {roc_auc_score(y_test,model.predict(X_test))}')
    print(f'F1 Score : {f1_score(y_test,model.predict(X_test))}')
    print(f'Precision : {precision_score(y_test,model.predict(X_test))}')
    print(f'Recall : {recall_score(y_test,model.predict(X_test))}')
    print('')
    print(conf_matrix)
    print('')

    #plot_confusion_matrix(model, X_test, y_test, normalize='true')

    return scores