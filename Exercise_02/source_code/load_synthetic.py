import random
import matplotlib.pyplot as plt 
from decision_tree import build_tree


def main():
    colors = ["#eaeaf2", "#721817", "#2b4162", "#fa9f42"]

    data_points = []
    for i in range(25):
        noiseA = random.uniform(-50, 50)
        noiseB = random.uniform(-50, 50)
        data_points.append( ( [  0+noiseA,   0+noiseB], colors[3]) )
        data_points.append( ( [100+noiseA,   0+noiseB], colors[1]) )
        data_points.append( ( [  0+noiseA, 100+noiseB], colors[2]) )
        data_points.append( ( [100+noiseA, 100+noiseB], colors[3]) )

    X = [a[0][0] for a in data_points]
    Y = [a[0][1] for a in data_points]
    D = [a[1] for a in data_points]
    # print( len(X), X[0] )
    # print( len(Y), Y[0]  )
    plt.scatter(X, Y, c=D)
    plt.savefig("dataset_synthetic.png", dpi=300, bbox_inches='tight')

    tree = build_tree(data_points, {0: "feature 1", 1: "feature 2"}, {})
    tree.print()
    x = [ 25.0, 125.0 ]
    prediction = tree.predict(x)
    print(prediction)

if __name__ == "__main__":
    main()