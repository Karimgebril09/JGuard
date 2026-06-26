import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd


def plot_hypothesis_test(t_stat, df, alpha=0.05,title="Two-Tailed T-Test"):
    """Plot the results of a two-tailed t-test """
    x= np.linspace(-5, 5, 1000)
    y= stats.t.pdf(x, df)

    t_critical= stats.t.ppf(1 - alpha/2, df)
    plt.figure(figsize=(10,5))
    plt.plot(x, y, linewidth=2)

    # rejection regions
    plt.fill_between(  x,  y,  where=(x <= -t_critical), alpha=0.3,label=f"alpha/2= {alpha/2:.3f}")

    plt.fill_between( x,  y, where=(x >= t_critical), alpha=0.3  )

    plt.axvline(-t_critical,linestyle="--", label=f"-t critical= {-t_critical:.2f}")
    plt.axvline( t_critical, linestyle="--",label=f"t critical= {t_critical:.2f}")
    plt.axvline( t_stat,color="red",linewidth=3,label=f"Observed t= {t_stat:.2f}")

    plt.title(title)

    plt.xlabel("t value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_bootstrap_distribution(
    bootstrap_diffs,
    observed_diff,
    lower,
    upper,
    title="Bootstrap Confidence Interval"
):
    """plot the bootstrap distribution with observed difference and confidence interval"""
    plt.figure(figsize=(10,5))
    plt.hist(bootstrap_diffs,bins=40,density=True,alpha=0.7)
    plt.axvline(observed_diff,color="red", linewidth=3, label=f"Observed= {observed_diff:.4f}" )
    plt.axvline(lower,linestyle="--",label=f"Lower CI= {lower:.4f}" )
    plt.axvline(upper,linestyle="--", label=f"Upper CI= {upper:.4f}")
    plt.axvline( 0,color="black", linestyle=":")
    plt.title(title)

    plt.xlabel("Mean Difference")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    
    
def plot_class_distributions(scores, labels, title="Distribution by Class", xlabel="Score"):
    """see tghe distribution of scores for each class"""
    class0= scores[labels== 0]
    class1= scores[labels== 1]

    plt.figure(figsize=(10, 5))

    plt.hist(class0, bins=30, alpha=0.6, density=True, label="Class 0")
    plt.hist(class1, bins=30, alpha=0.6, density=True, label="Class 1")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    
    
    
    
def plot_box_separation(scores, labels, class_names=("Class 0", "Class 1"), title="Boxplot Separation"):
    """plot boxplot for each class"""
    data0= scores[labels== 0]
    data1= scores[labels== 1]

    plt.figure(figsize=(7, 5))

    plt.boxplot([data0, data1], label=class_names)

    plt.title(title)
    plt.ylabel("Score")
    plt.grid(alpha=0.2)
    plt.show()

