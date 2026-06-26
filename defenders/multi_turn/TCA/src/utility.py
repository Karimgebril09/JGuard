
import numpy as np
from sklearn.preprocessing import PowerTransformer


def transform_feature(series, transform):
    if transform=="original":
        return series
    
    #for the values >0 and i care bout small values to be same and large values compressed 
    elif transform == "log1p":
        transformed=[]
        for value in series:

            if value < 0:
                value=0
            transformed.append(np.log(1 + value))
        return np.array(transformed)
    
    # here i want to emphasize large values and small values to be same
    elif transform=="square":

        transformed=[]

        for value in series:
            transformed.append(value * value)

        return np.array(transformed)

    # here i compress the info to be either 0 or 1
    elif transform=="binarize":

        transformed=[]
        for value in series:
            if value > 0:
                transformed.append(1.0)
            else:
                transformed.append(0.0)
        return np.array(transformed)

    #here convert to a more gaussian dirstribution handle negative values and zeros find best transformation 
    #is finding the best lambda automatically via maximum likelihood optimization
    elif transform=="yeo-johnson":

        transformer=PowerTransformer(method="yeo-johnson",standardize=False)
        values=np.asarray(series).reshape(-1, 1)
        transformed=transformer.fit_transform(values)
        return transformed.flatten()

    return series
