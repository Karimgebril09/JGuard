import numpy as np
from scipy import stats

def simple_two_sample_test(df, score_col="progressive_risk", label_col="label"):
    
    safe = df[df[label_col] == 0][score_col].values
    attack = df[df[label_col] == 1][score_col].values

    # means
    mean_safe =np.mean(safe)
    mean_attack= np.mean(attack)
    # variances
    var_safe = np.var(safe, ddof=1)
    var_attack =np.var(attack, ddof=1)

    # sample sizes
    n_safe =len(safe)
    n_attack = len(attack)

    #standard Error
    se =np.sqrt(var_safe /n_safe + var_attack/ n_attack)
    #t statistic
    t_stat = (mean_attack -mean_safe) / se

    #degrees of freedom 
    df_t = n_safe +n_attack - 2

    #p-value
    p_value = 2 *(1 - stats.t.cdf(abs(t_stat), df_t))

    #critical value
    t_critical =stats.t.ppf(0.975, df_t)
    #confidence interval
    lower= (mean_attack-mean_safe) -t_critical* se
    upper= (mean_attack- mean_safe)+t_critical *se

   
    print(f"Mean Safe : {mean_safe}")
    print(f"Mean Attack: {mean_attack}")
    print(f"Difference : {(mean_attack - mean_safe)}")
    print(f"Standard Error: {se}")
    print(f"t-statistic:{t_stat}")
    print(f"p-value:{p_value}")
    print(f"t-critical: {t_critical}")
    print(f"95% CI : [{lower}, {upper}]")

    if abs(t_stat) > t_critical:
        print("Decision: REJECT H0")
    else:
        print("Decision: FAIL TO REJECT H0")

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "ci": (lower, upper),
        "df": df_t
    }
    
    

def simple_bootstrap_test(df,feature_col="progressive_risk",label_col="label",n_resamples=5000):
    
    safe =df[df[label_col] == 0][feature_col].values
    attack =df[df[label_col] == 1][feature_col].values

    #sample means
    mean_safe=np.mean(safe)
    mean_attack = np.mean(attack)

    #observed difference
    diff = mean_attack - mean_safe

    #bootstrap
    bootstrap_diffs = []
    np.random.seed(42)
    for _ in range(n_resamples):
        boot_safe = np.random.choice(safe, size=len(safe), replace=True)
        boot_attack = np.random.choice(attack,size=len(attack),replace=True )
        bootstrap_diffs.append(np.mean(boot_attack) - np.mean(boot_safe))

    #95% confidence interval
    lower= np.percentile(bootstrap_diffs, 2.5)
    upper= np.percentile(bootstrap_diffs, 97.5)

    print(f"Mean Safe:{mean_safe}")
    print(f"Mean Attack:{mean_attack}")
    print(f"Difference:{diff}")
    print(f"95% CI: [{lower},{upper}]")

    if lower > 0 or upper < 0:
        print("Decision: REJECT H0")
        print("Conclusion: Significant difference between groups")
    else:
        print("Decision: FAIL TO REJECT H0")
        print("Conclusion: No significant difference detected")

    return diff, (lower, upper),bootstrap_diffs