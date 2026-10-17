This project was created by Srivibhav Amirisetty, one of the most distinguished science fair participants in Texas.
Srivibhav Amirisetty has won 2nd at the Greater Austin Regional Science and Engineering Fair, 1st at the Texas Regional Science and Engineering Fair, and has placed as a Top 300 Junior Innovator out of 65,000+ possible competitors in the Nation. This is the coding part of his award winning project. He hopes that it will inspire more people to compete and give people hope that Alzheimer's Disease Diagnosis can be solved in the near future.

This project is about Diagnosing Alzheimer's Disease using Explainable Machine Learning Models and EEG Data. 

There are two models featured in the code: Logistic Regression and Random Forests. If you want the code to compare the two, set the parameter "compare_rf" to True. If "compare_rf" is set to False, then the code will run only Logsitic Regression.
There are five features currently in the "self.features" list. Some others are also calculated. If you want to add them, type the name in the "self.features" list, the Box Plot dictionary, and the "expected_features" list.
The data is currently from a dataset of 88 patients; however, only 36 are AD, and 29 are healthy. There are eyes open and eyes closed datasets, consisting of the same patients.

The code generates box plots, SHAP bar charts, violin charts, and dot plots for each model. It also genereates a SHAP bar chart comparision if comparing the models. There is a side by side comparision as well. 

You can have the code do a tiny test (10 patients), a half-test (20 patients), and all the patients (65). You can use a flag to control which mode you want (--tiny, --test, and --full respectively).

