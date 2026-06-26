class LocalFeatures:
    def __init__(self, feature_name,oper):
        self.feature_name = feature_name
        self.oper = oper
    
    def compute(self,X,y_prev,y,i):
        return int(self.oper(X, y_prev, y, i))
    
class FeaturesManager:
    def __init__(self):
        self.features = []
    
    def add_feature(self,feature_name,oper):
        self.features.append(LocalFeatures(feature_name,oper))
    
    def names(self):
        return [feature.feature_name for feature in self.features]
    