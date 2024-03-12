# gn4pion (mjg_dev version)

## Note:

To run, you must have MLTree.root files with all topocluster moments.
First, put all the .root files in a directory, and transfer them to .npy files using the provided rootToNumpy.py script. You will need to specificy the input dir & the output dir, as well as the class it is (pi0,piplus,n0 etc. )

This will make .npy files, which you will point to the directory in  the config script you will be using.

## How to use
From a /run directory, you run the following:

```
python ../gn4pions_eastbay/train_mjg_piplus_pizero.py ../gn4pions_eastbay/gn4pions/configs/test_mjg_n0_pi0.yaml
```

## Evaluation Plots
After running the GNN, you will get a /results folder . Somewhere in that folder will be a file called predictions.npz . In plotter.py, make the save_path that directory. If you are unsure, see plotter.py for the example. Then run
```
python ../gn4pions_eastbay/plotter.py 
```
This will make:
 - Loss curve for both regression, classification and total 
 - Prediction vs Truth for both regression and classificiation 
 - ROC curve & confusion matrix for classification 
 - Response Plot for both classes (currently working for 2 classes)
