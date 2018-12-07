import matplotlib.pyplot as plt
import numpy as np


ensemble_MSE = [0.008673587937007598, 0.008797529599716873, 0.09883503932728671]
reg_MSE = [0.005504703152755579, 0.024069699053369944, 0.11209986164446104]

MSEs = [0.008673587937007598,0.005504703152755579,0.008797529599716873,0.024069699053369944,0.09883503932728671,0.11209986164446104]
x = [0,1,2,3,4,5]
x1 = np.array([0,1,2])
width = 0.35
ticks = ["ensemble train MSE", "regular train MSE", "ensemble validation MSE","regular validation MSE",
         "ensemble test MSE","regular test MSE"]
ticks2 = ["train", "val", "test"]

colors = ["b","r","b","r","b","r"]
plt.bar(x1,ensemble_MSE, width,color="r", label="ensemble")
plt.bar(x1+width,reg_MSE, width, color="b", label="regular")
plt.ylabel("MSE")
plt.title("Ensemble vs. Regular")
plt.xticks(x1+width,ticks2)
plt.legend(loc=2)

plt.savefig("ens_vs_reg.png", format="png")