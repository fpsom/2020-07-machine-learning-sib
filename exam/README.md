
# exam of Introduction to machine learning

The goal of this exam is perform an analysis on data related to heart disease,
in particular, we want to explore the relationship between a `target` variable - whether patient has a heart disease or not - and several other variables such as cholesterol level, age, ...

The data is present in the file `'heartData_simplified.csv'`, which is a cleaned and simplified version of the [UCI heart disease data set](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

We ask that you explore the data-set and answer the questions in a commented R code (or Rmd if you know how). 
You should send your code to **monique.zahn@sib.swiss** by the **7th of August**.

Do not hesitate to comment your code to explain to us your thought process and detail your conclusions following the analysis.



## description of the columns

* age : Patient age in years
* sex : Patient sex
* chol : Cholesterol level in mg/dl. 
* thalach : Maxium heart rate during the stress test
* oldpeak : Decrease of the ST segment during exercise according to the same one on rest.
* ca : Number of main blood vessels coloured by the radioactive dye. The number varies between 0 to 3.
* thal : Results of the blood flow observed via the radioactive dye.
	* fixed -> fixed defect (no blood flow in some part of the heart)
	* normal -> normal blood flow
	* reversible -> reversible defect (a blood flow is observed but it is not normal)
* target : Whether the patient has a heart disease or not

## code to read the data

```
heartData <- read.csv('heartData_simplified.csv')
heartData$target=as.factor(heartData$target)
heartData$sex=as.factor(heartData$sex)
heartData$thal=as.factor(heartData$thal)
```

## questions

1. perform a PCA on the `age`, `chol`, `thalach`, `ca` and `oldpeak` features. Do the PCA axes helps you to visually distinguish patients along different categorized features such as `target`, `sex` or `thal` ?

2. perform a Hierarchical Clustering on all features but `target`. Evaluate the quality of your clustering and explore the different options (distances, clustering method).

3. regression
	1. split your data-set in a train and test set.
	2. use a GLM to model and predict `target` using the other features with the train set
	3. evaluate your GLM model with the test set.
	4. which features are the most relevant to predict heart disease (`target`)

