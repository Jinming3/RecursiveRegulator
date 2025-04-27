The methods "Recursive Regulator" focuses on nonlinear systems modeling and real-time model adaptation.


# Installation requirements:

Language: Python 3.12

numpy 

torch 

matplotlib

scipy

# Folders

### head
callable functions and metrics

### RecursiveRegulator

System examples ready to run.

If a nonlinear model doesn't exist, 'train_{system}.py' is to train and save nonlinear model under static condition into folder "models".

'update_{system}.py' uses varying system data to test the recursive regulator for adapting trained model, e.g., update_RLC.py. It will produce images to show the regulator's performance in different system conditions.

The function path may problem, please check the path.
