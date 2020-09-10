import pandas as pd, matplotlib.pyplot as plt

def plot_history_accslosses():
    
    history = pd.read_csv("History.csv")
    
    loss_train = [a for a in history['loss']]
    loss_test = [b for b in history['val_loss']]
    accuracy_train = [c for c in history['acc']]
    accuracy_test = [d for d in history['val_acc']]
    
    plt.rcParams["figure.figsize"] = (18,12);
    fig, ax1 = plt.subplots()
    
    plt.title("Progressive History Loss & Accuracy Plot")
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(loss_train, color='red', linestyle=":"); ax1.plot(loss_test, color='blue', linestyle=":")
    ax1.tick_params(axis='y')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.set_ylabel('Accuracy')
    ax2.plot(accuracy_train, color='orange', linewidth=2); ax2.plot(accuracy_test, color='green', linewidth=2)
    ax2.tick_params(axis='y')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower left')
    
    def annotate_acc(y, ax=None):
        acc = max(y); epo = y.index(acc)
        text= "Best Model Saved Here! - Validation Accuracy: {:.3f}%".format(acc*100)
        if not ax: ax2=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.6", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax2.annotate(text, xy=(epo, acc), xytext=(0.94,0.96), **kw)
        return epo
    epo = annotate_acc(accuracy_test)
    
    ax1.annotate("  Respective Validation Loss: {:.3f}".format(loss_test[epo]), xy=(epo, loss_test[epo]),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    fig.tight_layout()