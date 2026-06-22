class RiskCalculator:
    def __init__(self, alpha=0.45, beta=0.25, gamma=0.30,inter_alpha=0.5, inter_beta=0.5,pattern_alpha=0.4, pattern_beta=0.6):
        self.alpha=alpha
        self.beta =beta
        self.gamma=gamma
        self.inter_alpha=inter_alpha
        self.inter_beta =inter_beta
        self.pattern_alpha=pattern_alpha
        self.pattern_beta =pattern_beta

    def compute_interaction_risk(self, features):
        #compute interaction between threat and toxicity in the conversation how attacker is trying to manipulate the conversation
        return (self.inter_alpha* features["threat_score"] +self.inter_beta * features["toxicity_score"] )*5

    def compute_pattern_risk(self, features):
        # calculate how much the conversation is drifting from the original topic and how fast it is drifting and interaction between them
        return (self.pattern_alpha* features["topic_drift_score"] +self.pattern_beta* features["drift_acceleration"])*5

    def calculate_progressive_risk(self, features, prev_progressive):
        # this is the final progressive risk keept in memory for the next turn to cacluate form it the next turn 
        interaction_risk=self.compute_interaction_risk(features)
        pattern_risk=self.compute_pattern_risk(features)

        progressive=(self.alpha* prev_progressive +self.beta * interaction_risk +self.gamma* pattern_risk)
        return progressive
