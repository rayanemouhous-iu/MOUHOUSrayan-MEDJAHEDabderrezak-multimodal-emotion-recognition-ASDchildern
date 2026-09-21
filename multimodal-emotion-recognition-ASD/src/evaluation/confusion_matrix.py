import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
def plot_confusion_matrix(y_true,y_pred,labels,class_names,normalize=None,title='Confusion matrix'):
    cm=confusion_matrix(y_true,y_pred,labels=labels,normalize=normalize); fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(cm,display_labels=class_names).plot(ax=ax,xticks_rotation=45,values_format='.2f' if normalize else 'd'); ax.set_title(title); fig.tight_layout(); return fig,ax
