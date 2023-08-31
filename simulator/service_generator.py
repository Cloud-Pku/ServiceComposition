from scipy import stats
import pickle
import numpy as np

class ServiceGenerator:
    def __init__(self, df_path = '/home/PJLAB/chenyun/ServiceComposition/simulator/dataset/qws2/qws_df.pkl'):
        with open(df_path, 'rb') as f:
            self.df = pickle.load(f)
        self.attributes = self.df.columns
        self.data = {}
        self.bounds = {}
        self.mean = {}
        self.std = {}
        self.kdes = {}
        for key in self.attributes[:9]:
            self.data[key] = self.df[key].values
            self.bounds[key] = [min(self.data[key]), max(self.data[key])]
            self.mean[key] = np.mean(self.data[key])
            self.std[key] = np.std(self.data[key])
            self.kdes[key] = stats.gaussian_kde(self.data[key])
        self.bounds['Availability'] = [0., 1.]
        self.bounds['Successability'] = [0., 1.]
        self.bounds['Reliability'] = [0., 1.]
        print('complete')
        
    def sample(self, nums=1, attributes=['Response Time', 'Availability', 'Throughput', 'Reliability']):
        services = [{}] * nums
        for att in attributes:
            tmp_atts = self.kdes[att].resample(nums)[0]
            np.clip(tmp_atts, self.bounds[att][0], self.bounds[att][1])
            for i in range(nums):
                services[i][att] = tmp_atts[i]
        return services


if __name__ == '__main__':
    gen = ServiceGenerator()
    print(gen.sample(9))

    # new_samples = stats.gaussian_kde(data)(np.linspace(-2, 12, num=1000)).round(decimals=2)