The methods "Recursive Regulator" focuses on nonlinear systems modeling and real-time model adaptation.

Related article>>
Recursive regulator: a deep-learning and real-time model adaptation strategy for nonlinear systems.

https://doi.org/10.1038/s44172-025-00477-4


# Installation requirements:

Language: Python 3.11

# Method and demonstration
### Design

The method's functions are in file "header" and "pem".

### Data and demonstration 

In folder RecursiveRegulator, there are system examples ready to run.

'update_{system}.py' generates varying system data to test the recursive regulator for adapting trained model. For example, update_EMPS.py, will produce images to show the regulator's performance in different system conditions.

If a nonlinear model doesn't exist when run "update.py", 'train_{system}.py' is to train and save nonlinear model under static condition into folder "models".



