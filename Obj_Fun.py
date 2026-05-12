import numpy as np
from Global_Vars import Global_Vars
from Model_H_DA_AD_DN import Model_H_DA_AD_DN

def Obj_fun_pred(Soln):
    Feat1 = Global_Vars.Feat1
    Feat2 = Global_Vars.Feat2
    Target = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Eval = Model_H_DA_AD_DN(Feat1, Feat2, Target, sol)
        Fitn[i] =(1 /Eval[0]) +Eval[6] # Maximization of Accuracy and minimization of MSE
    return Fitn

