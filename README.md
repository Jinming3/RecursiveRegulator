The methods "Recursive Regulator" focuses on nonlinear systems modeling and real-time model adaptation.
Related article>>
Recursive regulator: a deep-learning and real-time model adaptation strategy for nonlinear systems.
https://doi.org/10.1038/s44172-025-00477-4


# Installation requirements:

Language: Python 3.12

# Folders

### RecursiveRegulator

System examples ready to run.

If a nonlinear model doesn't exist, 'train_{system}.py' is to train and save nonlinear model under static condition into folder "models".

'update_{system}.py' uses varying system data to test the recursive regulator for adapting trained model, e.g., update_EMPS.py. It will produce images to show the regulator's performance in different system conditions.

The function path may problem, please check the path.
