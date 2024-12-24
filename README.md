# Whats-Cooking-Predicting-Cuisine.
![Main display - Home](https://github.com/user-attachments/assets/856bd45f-1972-443b-bd6d-a23d14954392)


# Introduction
The "Whats-Cooking-Predicting-Cuisine Application" leverages machine learning to predict the type of cuisine based on a list of ingredients. This application includes a user-friendly GUI developed using Tkinter and integrates various machine learning models for accurate predictions. It provides insightful accuracy comparisons and supports user inputs to predict cuisines interactively.

# Features
  -	Multi-Model Predictions: Implements models like KNN, Logistic Regression, Naive Bayes, SGD Classifier, and Random Forest for cuisine prediction.
  -	GUI Interface: Interactive and easy-to-use graphical interface for model selection and prediction display.
  -	Data Visualization: Accuracy graph for comparing model performance.
  -	Custom User Input: Allows users to input ingredients and choose a prediction model.
# Workflow
  1.	Data preprocessing using lemmatization and TF-IDF vectorization.
  2.	Training and testing multiple machine learning models.
  3.	Generating predictions and accuracy metrics for each model.
  4.	Displaying results via a Tkinter-based GUI.
# Tools and Technologies
  -	**Programming Language:** Python
  -	**Development Environment:** Visual Studio Code
  - **Key Libraries:** 
      *	GUI Development: Tkinter, PIL (Python Imaging Library)
      *	Data Processing: pandas, numpy, sklearn, nltk
      *	Machine Learning: scikit-learn
      *	Visualization: matplotlib
# Key Components
  -	**Preprocessing**: Combines ingredient lists into sentences and applies TF-IDF vectorization.
  -	**ML Models:** KNN, Logistic Regression, Multinomial Naive Bayes, SGD Classifier, and Random Forest.
  - **GUI Features**: 
      1.	Model accuracy display.
      2.	Real-time predictions from user inputs.
      3.	Graphical comparison of model performance.
# Usage
  1.	Launch the application by running the Python script in Visual Studio Code.
  2.	Choose a prediction model from the GUI.
  3.	Input ingredient lists for prediction.
  4.	View prediction results and accuracy metrics.

![User Input](https://github.com/user-attachments/assets/5e223f6e-21e6-4de0-be7e-b0486cf8c235)


# Results
  -	The application demonstrates varying accuracy levels across models, with results saved as CSV files for further analysis.
  -	Accuracy comparison graph visualizes the performance of all implemented models.

- **Classification Report**
![Report analysis](https://github.com/user-attachments/assets/42cea1a6-c58a-44fe-9181-12d26802d533)


- **Accuracy Graph**

![Accuracy Graph](https://github.com/user-attachments/assets/5c2a07e5-7c41-4938-a170-bb4c7bdc342d)


