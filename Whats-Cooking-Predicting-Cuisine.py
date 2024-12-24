import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import pandas as pd
from pandas import DataFrame
import numpy as np
import ast
import csv
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#####################################  Path of Training and Testing Files   ##########################################
rf = r" \train.json"
rf_test = r" \test.json"
#######################################   Reading Data Files  ###########################################################
training = pd.read_json(rf)      
testing = pd.read_json(rf_test)       
print("File Reading Completed.\n")
#######################################   Saving Testing ID  ###########################################################
test_id_list = testing["id"].tolist()

def string_com(x):
    ans = ' '.join(x)
    ans = lemmatizer.lemmatize(ans)
    return ans
######################################   Ingredients combine and saving as sentence  ####################################
training["ingredients"] = training["ingredients"].apply(string_com)
testing["ingredients"] = testing["ingredients"].apply(string_com)

######################################   Tf_Idf Vectorizer   ########################################################### 
tf_idf = TfidfVectorizer()
tf_idf.fit(training["ingredients"])
train_data_vect = tf_idf.transform(training["ingredients"])
test_data_vect = tf_idf.transform(testing["ingredients"])

X_train,X_test,y_train,y_test = train_test_split(train_data_vect,training["cuisine"],random_state=0,train_size=0.8)

def file_write(path, dict1):
    file1 = open(path,"w")
    writer = csv.writer(file1)
    writer.writerow(["id", "cuisine"])
    for key, value in dict1.items():
        writer.writerow([key, value])
    file1.close()

def dict_of_models(list_cus):
    dict_cus = {}
    i = 0
    for idc in test_id_list:
        dict_cus[idc] = list_cus[i]
        i += 1
    return dict_cus

window = tk.Tk()
window.title(" Whats Cooking? Predicting Cuisine. ")
display = tk.Canvas(window, width=800, height=500)      #### Creating GUI window #####
display.pack()

image = Image.open(r" \try4.jpeg")
image = image.resize((800,800), Image.LANCZOS) 
img = ImageTk.PhotoImage(image)      ## Image loading ##
imglabel = tk.Label(window, image=img)
display.create_window(400, 200, window=imglabel)

label1 = tk.Label(window, text='Whats Cooking? Predicting Cuisine.')
label1.config(font=('Arial', 16))
display.create_window(400, 40, window=label1)

##########################################             KNN Model                 ####################################
from sklearn import metrics                                     
from sklearn.metrics import classification_report

def knn_pred_func(testing_vector):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred_test = knn.predict(X_test)
    final_predict = knn.predict(testing_vector)          ###############    Final KNN   #####################
    # print("\n Actual Predict\n",final_predict)
    acc_knn = (metrics.accuracy_score(y_test, pred_test))*100
    # print("Accuracy KNN Model:",acc_knn)
    prec_rec_knn = classification_report(y_test, pred_test)
    # print("Precision and Recall:\n", classification_report(y_test, pred_test))
    return final_predict, acc_knn, prec_rec_knn

final_predict, acc_knn, prec_rec_knn = knn_pred_func(test_data_vect)
#################################   Saving Output in File    #######################################
dict_id_cuisine = dict_of_models(final_predict)
path = r' \output_knn.csv'
file_write(path, dict_id_cuisine)
print("Result written in Output File(KNN)")
print("\n********************************************************************************\n")
def knn_btn():
    label_knn = tk.Label(window, text=acc_knn) 
    display.create_window(70, 250, window=label_knn)

#####################################            Logistic Regression             ###################################
from sklearn.linear_model import LogisticRegression
def log_pred_func(testing_vector):
    logreg = LogisticRegression(C=3)                              ###########   Increasing Accuracy by decreasing C value  ##########
    logreg.fit(X_train,y_train)
    logistic_pred=logreg.predict(X_test)                          
    log_predict = logreg.predict(testing_vector)                  #############  Final List  Logistic  ###################
    # print("\n Logictic regression Predict\n",log_predict)
    acc_log = (metrics.accuracy_score(y_test, logistic_pred))*100
    # print("\nAccuracy Logistic Model:", acc_log)
    prec_rec_log = classification_report(y_test, logistic_pred)
    # print("Precision and Recall:\n", classification_report(y_test, logistic_pred))
    return log_predict, acc_log, prec_rec_log

#################################   Saving Output in File    #######################################
log_predict, acc_log, prec_rec_log = log_pred_func(test_data_vect) 
dict_id_cuisine_log_regr = dict_of_models(log_predict)
path_log = r' \output_log_reg.csv'
file_write(path_log, dict_id_cuisine_log_regr)
print("Result written in Output File(log_reg)")
print("\n********************************************************************************\n")
def log_btn():
    label_log = tk.Label(window, text=acc_log) 
    display.create_window(270, 250, window=label_log)

##############################                    NAIVE BAYES(MULTINOMIAL)               ###################################
# from sklearn.naive_bayes import GaussianNB   #############   accuracy 24% approx  ##########
from sklearn.naive_bayes import MultinomialNB  #############   accuracy 68% approx  ##########
def nb_pred_func(testing_vector):
    NB = MultinomialNB(alpha=0.5)
    NB.fit(X_train.toarray(), y_train)
    naive_bayes = NB.predict(X_test.toarray())
    naive_bayes_predict = NB.predict(testing_vector.toarray())                ############  Final Naive Bayes  #########################
    acc_naive = (metrics.accuracy_score(y_test, naive_bayes))*100
    # print("Accuracy of Naive Bayes:",acc_naive)
    prec_rec_nb = classification_report(y_test, naive_bayes)
    # print("Precision and Recall:\n", classification_report(y_test, naive_bayes))
    return naive_bayes_predict, acc_naive, prec_rec_nb

naive_bayes_predict, acc_naive, prec_rec_nb = nb_pred_func(test_data_vect)
#################################   Saving Output in File    #######################################
dict_id_cuisine_naive = dict_of_models(naive_bayes_predict)
path_naive = r' \output_naive.csv'
file_write(path_naive, dict_id_cuisine_naive)
print("Result written in Output File(naive)")
print("\n********************************************************************************\n")

def nb_btn():
    label_nb = tk.Label(window, text=acc_naive)
    display.create_window(470, 250, window=label_nb)

####################################                SGD CLASSIFIER                ##########################################
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
def sgd_pred_func(testing_vector):
    sgd_clf = linear_model.SGDClassifier()
    sgd_clf.fit(X_train, y_train)
    SGD_train = sgd_clf.predict(X_test)
    SGD_final_predict = sgd_clf.predict(testing_vector)                       ############  Final SGD  #########################
    acc_sgd = (metrics.accuracy_score(y_test, SGD_train))*100
    # print("Accuracy of SGD classifier:",acc_sgd)
    prec_rec_sgd = classification_report(y_test, SGD_train)
    # print("Precision and Recall:\n", classification_report(y_test, SGD_train))
    return SGD_final_predict, acc_sgd, prec_rec_sgd

#################################   Saving Output in File    #######################################
SGD_final_predict, acc_sgd, prec_rec_sgd = sgd_pred_func(test_data_vect)
dict_id_cuisine_SGD = dict_of_models(SGD_final_predict)
path_SGD = r' \output_SGD.csv'
file_write(path_SGD, dict_id_cuisine_SGD)
print("Result written in Output File(SGD)")
print("\n********************************************************************************\n")
def sgd_btn():
    label_sgd = tk.Label(window, text=acc_sgd)
    display.create_window(670, 250, window=label_sgd)

######################################            RANDOM_FOREST CLASSIFIER            ###########################################
from sklearn.ensemble import RandomForestClassifier
def rand_pred_func(testing_vector):
    rand_for_fit = RandomForestClassifier(max_depth=60, n_estimators=20).fit(X_train,y_train)
    rand_forest_train = rand_for_fit.predict(X_test)
    rand_forest_final_predict = rand_for_fit.predict(testing_vector)                 ############  Final Random Forest  #############
    acc_rand = (metrics.accuracy_score(y_test, rand_forest_train))*100
    # print("Accuracy of Random_Forest classifier:",acc_rand)
    prec_rec_rf = classification_report(y_test, rand_forest_train)
    # print("Precision and Recall:\n", classification_report(y_test, rand_forest_train))
    return rand_forest_final_predict, acc_rand, prec_rec_rf

#################################   Saving Output in File    #######################################
rand_forest_final_predict, acc_rand, prec_rec_rf = rand_pred_func(test_data_vect)
dict_id_cuisine_rand_forest = dict_of_models(rand_forest_final_predict)
path_rand = r' \output_rand_forest.csv'
file_write(path_rand, dict_id_cuisine_rand_forest)
print("Result written in Output File(rand_forest)")
print("\n********************************************************************************\n")
def rf_btn():
    label_rf = tk.Label(window, text=acc_rand)
    display.create_window(250, 370, window=label_rf)

def graph():
    x = ["KNN", "Logistic Reg", "Naive Bayes", "SGD", "Random Forest"]
    y = [acc_knn, acc_log, acc_naive, acc_sgd, acc_rand]
    plt.bar(x,y)
    plt.xlabel(' Models For Predicting Cuisines')
    plt.ylabel('Accuracy')
    plt.title(' Whats Cooking? Predicting Cuising') 
    plt.show()

label_head_knn = tk.Label(window, text=' KNN Model ', font=('Arial', 13))   
display.create_window(70, 180, window=label_head_knn)

button_knn = tk.Button(text='Show', width=8, command=knn_btn)
display.create_window(70, 220, window=button_knn)

label_head_log = tk.Label(window, text=' Logistic Regression ', font=('Arial', 13))
display.create_window(270, 180, window=label_head_log)

button_log = tk.Button(text='Show', width=8, command=log_btn)
display.create_window(270, 220, window=button_log)

label_head_nb = tk.Label(window, text=' Naive Bayes(Multinomial) ', font=('Arial', 13)) 
display.create_window(470, 180, window=label_head_nb)

button_nb = tk.Button(text='Show', width=8, command=nb_btn)
display.create_window(470, 220, window=button_nb)

label_head_sgd = tk.Label(window, text=' SGD Classifier ', font=('Arial', 13))  
display.create_window(670, 180, window=label_head_sgd)

button_sgd = tk.Button(text='Show', width=8, command=sgd_btn)
display.create_window(670, 220, window=button_sgd)

label_head_rf = tk.Label(window, text=' Random Forest ', font=('Arial', 13))    
display.create_window(250, 310, window=label_head_rf)

button_rf = tk.Button(text='Show', width=8, command=rf_btn)
display.create_window(250, 340, window=button_rf)

label_head_graph = tk.Label(window, text=' Accuracy Graph of models ', font=('Arial', 13))     
display.create_window(450, 310, window=label_head_graph)

button_graph = tk.Button(text='Show', width=8, command=graph)
display.create_window(450, 340, window=button_graph)

def createNewWindow():
    global my_img
    newWindow = tk.Toplevel(window)
    display = tk.Canvas(newWindow, width=800, height=500)
    display.pack()
    my_img = ImageTk.PhotoImage(Image.open(r" \try4.jpeg"))
    my_label = tk.Label(newWindow, image=my_img)#.pack()
    display.create_window(400, 200, window=my_label)
    label4 = tk.Label(newWindow, text=' Whats Cooking? Predicting Cuisine. ', font=('Arial', 16))
    label4.place(relx=0.35,rely=0.06)

    label_knn_new = tk.Label(newWindow, text=' KNN Model ', font=('Arial', 13))
    label_knn_new.place(relx=0.1,rely=0.16)

    label_knn = tk.Label(newWindow, text=prec_rec_knn)
    label_knn.place(relx=0.04,rely=0.2)

    label_head_log = tk.Label(newWindow, text=' Logistic Regression ', font=('Arial', 13))   
    label_head_log.place(relx=0.26,rely=0.16)

    label_log = tk.Label(newWindow, text=prec_rec_log)                                       
    label_log.place(relx=0.22,rely=0.2)

    label_head_nb = tk.Label(newWindow, text=' Naive Bayes ', font=('Arial', 13))     
    label_head_nb.place(relx=0.46,rely=0.16)

    label_nb = tk.Label(newWindow, text=prec_rec_nb)
    label_nb.place(relx=0.40,rely=0.2)

    label_head_sgd = tk.Label(newWindow, text=' SGD Classifier ', font=('Arial', 13)) 
    label_head_sgd.place(relx=0.62,rely=0.16)

    label_sgd = tk.Label(newWindow, text=prec_rec_sgd)
    label_sgd.place(relx=0.58,rely=0.2)

    label_head_rf = tk.Label(newWindow, text=' Random Forest ', font=('Arial', 13))   
    label_head_rf.place(relx=0.8,rely=0.16)

    label_rf = tk.Label(newWindow, text=prec_rec_rf)                                  
    label_rf.place(relx=0.76,rely=0.2)

button_report = tk.Button(window, text="Show Classification Report", command=createNewWindow)
display.create_window(350, 440, window=button_report)
import tkinter.ttk as ttk
from tkinter.ttk import *

def user_input():
    global my_img
    userwindow = tk.Toplevel(window)
    display = tk.Canvas(userwindow, width=800, height=500)
    display.pack()
    my_img = ImageTk.PhotoImage(Image.open(r" \try4.jpeg"))
    my_label = tk.Label(userwindow, image=my_img)
    display.create_window(400, 200, window=my_label)
    label4 = tk.Label(userwindow, text=' Whats Cooking? Predicting Cuisine. ', font=('Arial', 16))
    label4.place(relx=0.35,rely=0.06)

    ListIng=[]
    def callback(event):
        ListIng.append(entryUnique.get())
        entryUnique.delete(0, 'end')
    OPTIONS = ["KNN","KNN","Logistic Regression","Naive Bayes","SGD Classifier","Random Forest"] 
    val = StringVar(userwindow)
    val.set(OPTIONS[0])
    # opt = ttk.OptionMenu(userwindow, val, "KNN","KNN""Logistic Regression","Naive Bayes","SGD Classifier","Random Forest")

    opt = OptionMenu(userwindow, val, *OPTIONS)
    opt.place(relx=0.45,rely=0.4)
    bar=Progressbar(userwindow,orient=HORIZONTAL,length=100,mode='determinate')
    def predict_user():
        ListIng = string_com(testing)
        ListIng = [ListIng]
        testing_ListIng = tf_idf.transform(ListIng)        
        bar['value']=20
        userwindow.update_idletasks()
        time.sleep(1)
        bar['value']=50
        userwindow.update_idletasks()
        time.sleep(1)
        bar['value']=80
        userwindow.update_idletasks()
        time.sleep(1)
        bar['value']=100
        if val.get() == "KNN":
            answer, x, y = knn_pred_func(testing_ListIng)
            label_ans_knn = tk.Label(userwindow, text=answer)
            label_ans_knn.config(font=('Arial', 12))
            label_ans_knn.place(relx=0.42,rely=0.6)
        if val.get() == "Logistic Regression":
            answer2, x, y = log_pred_func(testing_ListIng)
            label_ans_log = tk.Label(userwindow, text=answer2)
            label_ans_log.config(font=('Arial', 12))
            label_ans_log.place(relx=0.42,rely=0.65)
        if val.get() == "Naive Bayes":
            answer3, x, y = nb_pred_func(testing_ListIng)
            label_ans_nb = tk.Label(userwindow, text=answer3)
            label_ans_nb.config(font=('Arial', 12))
            label_ans_nb.place(relx=0.42,rely=0.7)
        if val.get() == "SGD Classifier":
            answer4, x, y = sgd_pred_func(testing_ListIng)
            label_ans_sgd = tk.Label(userwindow, text=answer4)
            label_ans_sgd.config(font=('Arial', 12))
            label_ans_sgd.place(relx=0.42,rely=0.75)
        if val.get() == "Random Forest":
            answer5, x, y = rand_pred_func(testing_ListIng)
            label_ans_rand = tk.Label(userwindow, text=answer5)
            label_ans_rand.config(font=('Arial', 12))
            label_ans_rand.place(relx=0.42,rely=0.8)
    bar.place(relx=0.42,rely=0.56)
    button_ok = Button(userwindow, text="Predict", command=predict_user)
    button_ok.place(relx=0.42,rely=0.52)

    label_knn = tk.Label(userwindow, text='Enter Ingredient:')
    label_knn.config(font=('Arial', 14))
    label_knn.place(relx=0.4,rely=0.2)

    entryUnique=tk.Entry(userwindow)
    entryUnique.place(relx=0.4,rely=0.25)
    label_select = tk.Label(userwindow, text='Select Desired Model For Prediction')
    label_select.config(font=('Arial', 13))
    label_select.place(relx=0.37,rely=0.35)

    userwindow.bind('<Return>', callback)

button_input_cusine = tk.Button(window, text="Wants to know your DESIRED CUISINE????", command=user_input)
display.create_window(350, 490, window=button_input_cusine)
window.mainloop()