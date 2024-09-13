The identification methods "Recursive Regulator" focuses on nonlinear systems and real-time model adaptation.


# Installation requirements:

Language: Python 3.10

numpy 

torch 

matplotlib

scipy

statsmodels

# Folders

### head
callable functions and metrics

### RecursiveRegulator

System examples ready to run.

If nonlinear model doesn't exist, 'train_{system}.py' is to train and save nonlinear model of static systems into folder "models".

'update_{system}.py' uses changing system data to test the recursive regulator for adapting trained model, e.g., update_RLC.py. It will produce images to show the regulator's performance in different system conditions.

If the package path had problem, please change the path to obsolute, e.g., "F:/head...". 
