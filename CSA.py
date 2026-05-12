import time
import numpy as np

def Sphere(x):
    return np.sum(x**2)

def bound(x, UB, LB):
    x = np.minimum(x, UB)
    x = np.maximum(x, LB)
    return x

def CSA(Crystal,ObjFunc,LB,UB, MaxIteration):
    Var_Number,Cr_Number = Crystal.shape
    Fun_eval = np.array([ObjFunc(ind) for ind in Crystal])

    # Best Crystal
    BestFitness = np.min(Fun_eval)
    idbest = np.argmin(Fun_eval)
    Crb = Crystal[idbest, :].copy()
    BestCr = Crb.copy()

    Eval_Number = Cr_Number
    Conv_History = [BestFitness]

    # ================= Main Loop =================
    Iter = 1
    ct = time.time()
    while Iter < MaxIteration:
        for i in range(Cr_Number):
            # Main Crystal
            Crmain = Crystal[np.random.randint(Cr_Number), :]

            # Random-selected Crystals
            RandNumber = np.random.randint(1, Cr_Number+1)  # at least 1
            RandSelectCrystal = np.random.choice(Cr_Number, RandNumber, replace=False)

            # Mean of randomly-selected Crystals
            if len(RandSelectCrystal) != 1:
                Fc = np.mean(Crystal[RandSelectCrystal, :], axis=0)
            else:
                Fc = Crystal[RandSelectCrystal[0], :]

            # Random numbers (-1,1)
            r = 2*np.random.rand() - 1
            r1 = 2*np.random.rand() - 1
            r2 = 2*np.random.rand() - 1
            r3 = 2*np.random.rand() - 1

            # New Crystals
            NewCrystal = np.zeros((4, Var_Number))
            NewCrystal[0, :] = Crystal[i, :] + r * Crmain
            NewCrystal[1, :] = Crystal[i, :] + r1 * Crmain + r2 * Crb
            NewCrystal[2, :] = Crystal[i, :] + r1 * Crmain + r2 * Fc
            NewCrystal[3, :] = Crystal[i, :] + r1 * Crmain + r2 * Crb + r3 * Fc

            # Evaluate new solutions
            for i2 in range(4):
                NewCrystal[i2, :] = bound(NewCrystal[i2, :], UB, LB)
                Fun_evalNew = ObjFunc(NewCrystal[i2, :])

                if Fun_evalNew < Fun_eval[i]:
                    Fun_eval[i] = Fun_evalNew
                    Crystal[i, :] = NewCrystal[i2, :]

                Eval_Number += 1

        # Best Crystal of iteration
        BestFitness = np.min(Fun_eval)
        idbest = np.argmin(Fun_eval)
        Crb = Crystal[idbest, :].copy()
        BestCr = Crb.copy()

        Conv_History.append(BestFitness)

        Iter += 1
    ct=time.time()-ct
    return BestFitness,Conv_History,BestCr,ct

