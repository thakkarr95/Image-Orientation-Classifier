# Implementing Machine Learning Algorithms to predice the orientation of Images

3 Classification Algorithms were implemented to predict the orientation of given images:

1. KNN Classifier
2. Random Forest Classifier
3. Adaboost Classifier
4. Best (The best classifier from the above 3 algorithms was KNN)

`orient.py` contains the complete code which implementes the algorithms stated above.

Usage:

To train a model (<model_name> can be nearest, adaboost, forest, best)

    $ python3 orient.py train train-data.txt <model_file>.pkl <model_name>

To test a model (<model_name> can be nearest, adaboost, forest, best)
   
    $ python3 orient.py test test-data.txt <model_file>.pkl <model_name>

Check `Problem Description.pdf` for more details related to problem description

The file `Final Report.pdf` contains detailed analysis of the results obtained from the 3 algorthims implemented.
