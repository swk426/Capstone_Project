
def training_results_Loss(results, title, dpi=200):
    import matplotlib.pyplot as plt
    """ results: the fitted model
        title: Title of the image to be saved
        dpi: default set at 300
        This function will generate Loss graph of fitted model using history and save pictures."""
    history = results.history
    plt.figure(figsize=(7,7))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(title, dpi=dpi)
    plt.show()

if __name__ == "__training_results_Loss__":
    train_results_Loss()
    
def training_results_Accuracy(results, title, dpi=200):
    import matplotlib.pyplot as plt

    """ results: the fitted model
    title: Title of the image to be saved
    dpi: default set at 300
    This function will generate Accuracy graph of fitted model using history and save pictures."""
    history = results.history
    plt.figure(figsize=(7,7))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(title, dpi=dpi)
    plt.show()

if __name__ == "__training_results_Accuracy__":
    train_results_Accuracy(results, title, dpi=200)

def cm_df(data, index, columns):
    import pandas
    df = pandas.DataFrame(data=data,index=index, columns=columns)
    return df

if __name__ == "cm_df":
    cm_df(data, index, columns)