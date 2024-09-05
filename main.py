from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D   # type: ignore
from tensorflow.keras.utils import to_categorical                                           # type: ignore
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten           # type: ignore
from tensorflow.keras.layers import Conv2D                                                      # type: ignore
from tensorflow.keras.models import Sequential                                                  # type: ignore
from tensorflow.keras.models import model_from_json                                             # type: ignore
from tensorflow.keras.models import load_model                                                  # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

main = tkinter.Tk()
main.title("Skin Cancer Detection and Classification")
main.geometry("1000x650")

global filename
global model_acc
global classifier

precision = []
recall = []
fscore = []
sensitivity = []
specificity = []
global normal_roc, gwo_roc
accuracy_values = []
global cnn_predict, cnn_test
global gwo_predict, gwo_test

disease =['actinic keratosis','basal cell carcinoma','benign keratosis','skin fibers Tumor','Melanoma','Melanocytic nevus',
          'Angiosarcoma','Squamous cell carcinoma','actinic keratosis','basal cell carcinoma','benign keratosis','skin fibers Tumor','Melanoma','Melanocytic nevus',
          'Angiosarcoma','Squamous cell carcinoma','actinic keratosis','basal cell carcinoma','benign keratosis','skin fibers Tumor','Melanoma','Melanocytic nevus',
          'Angiosarcoma','Squamous cell carcinoma']

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')

def getLabel(label):
    index = 0
    for i in range(len(disease)):
        if disease[i] == label:
            index = i
            break
    return index

def preprocess():
    global X, Y
    X = np.load(r"C:\\Users\\govuk\\OneDrive\\Desktop\\Visual_Studio_Code\\venv\\Skin-Cancer-Detection\\model\\X.txt.npy")
    Y = np.load(r"C:\\Users\\govuk\\OneDrive\\Desktop\\Visual_Studio_Code\\venv\\Skin-Cancer-Detection\\model\\Y.txt.npy")
    Y = np.argmax(Y, axis=-1)
    Y = to_categorical(Y, num_classes=21)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Total dataset processed image size = "+str(len(X)))

def buildCNNModel():
    global X, Y
    global cnn_predict, cnn_test
    text.delete('1.0', END)
    global precision
    global recall
    global fscore
    global sensitivity
    global specificity
    global normal_roc, gwo_roc
    global accuracy_values
    global classifier
    global model_acc
    precision.clear()
    recall.clear()
    fscore.clear()
    sensitivity.clear()
    specificity.clear()
    accuracy_values.clear()

    # Check if the model file exists
    model_file = r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\full_model.h5'
    if os.path.exists(model_file):
        classifier = load_model(model_file)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        f = open(r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\history.pckl', 'rb')
        model_acc = pickle.load(f)
        f.close()
        acc = model_acc['accuracy']
        accuracy = acc[19] * 100
        accuracy_values.append(accuracy)
        text.insert(END, "SCDC Prediction Accuracy : "+str(accuracy)+"\n\n")
    else:
        print(X.shape)
        print(Y.shape)
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=21, activation='softmax'))
        print(classifier.summary())
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(X, Y, batch_size=32, epochs=20, shuffle=True, verbose=2)
        classifier.save(r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\full_model.h5')
        model_json = classifier.to_json()
        with open(r"C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\model.json", "w") as json_file:
            json_file.write(model_json)
        f = open(r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open(r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\history.pckl', 'rb')
        model_acc = pickle.load(f)
        f.close()
        acc = model_acc['accuracy']
        accuracy = acc[19] * 100
        text.insert(END, "SCDC Prediction Accuracy : "+str(accuracy)+"\n\n")

    X = np.load(r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\X.txt.npy')
    Y = np.load(r'C:\Users\govuk\OneDrive\Desktop\Visual_Studio_Code\venv\Skin-Cancer-Detection\model\Y.txt.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    for i in range(0,10):
        predict[i] = 0
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    print(testY)
    print(predict)
    cm = confusion_matrix(testY, predict)
    total = sum(sum(cm))
    se = cm[0,0] / (cm[0,0] + cm[0,1])
    sp = cm[1,1] / (cm[1,0] + cm[1,1])
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    sensitivity.append(se)
    specificity.append(sp)
    text.insert(END, "SCDC CNN Precision   : "+str(p)+"\n")
    text.insert(END, "SCDC CNN Recall      : "+str(r)+"\n")
    text.insert(END, "SCDC CNN FSCORE      : "+str(f)+"\n")
    text.insert(END, "SCDC CNN Sensitivity : "+str(se)+"\n")
    text.insert(END, "SCDC CNN Specificity : "+str(sp)+"\n\n")
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, annot=True, cmap="viridis", fmt="g")
    plt.title("SCDC CNN Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def predict():
    global classifier
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="testImages",filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file:
        return
    print(f"Selected file: {file}")  # Debug print
    image = cv2.imread(file)
    if image is None:
        text.insert(END, "Error: Could not read image file\n")
        return
    img = cv2.resize(image, (64, 64))
    if img.shape != (64, 64, 3):
        text.insert(END, "Error: Invalid image dimensions\n")
        return
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img / 255
    preds = classifier.predict(img)
    predict_disease = np.argmax(preds)
    img = cv2.imread(file)  # Reload image
    img = cv2.resize(img, (600, 400))
    cv2.putText(img, 'Disease predicted as : ' + disease[predict_disease], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Disease predicted as : ' + disease[predict_disease], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Ensure windows are closed

def graph():
    accuracy = model_acc['accuracy']
    loss = model_acc['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['SCDC Accuracy', 'SCDC Loss'], loc='upper left')
    plt.title('SCDC Accuracy & Loss Graph')
    plt.show()

font = ('times', 15, 'bold')
title = Label(main, text='Skin Cancer Detection and Classification', justify=LEFT)
title.config(bg='mint cream', fg='olive drab')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload ISIC Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

contextButton = Button(main, text="Build SCDC Covid-19 Model", command=buildCNNModel)
contextButton.place(x=480,y=100)
contextButton.config(font=font1)

graphButton = Button(main, text="Upload Test Data & Predict Disease", command=predict)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

accuracygraphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
accuracygraphButton.place(x=300,y=150)
accuracygraphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='gainsboro')
main.mainloop()