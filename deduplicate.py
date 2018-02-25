import string
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sklearn.cluster
import jellyfish

class Deduplicate:
    def __init__(self):
        self.dobs = []
        self.fns = []
        self.lns = []
        self.genders = []
        self.names = []
        self.final_list = []
        self.cluster_num = 0

    def load_preprocess_data(self):
        data = pd.read_csv('data.csv')
        for idx in range(data.ln.shape[0]):
             fn = data.fn[idx].lower()
             ln = data.ln[idx].lower()
             self.fns.append(fn)
             self.lns.append(ln)
             fn_c = string.replace(fn,' ','')
             ln_c = string.replace(ln,' ','')
             dob = data.dob[idx]
             gender = data.gn[idx]
             self.names.append(fn_c.decode('utf-8') + ln_c.decode('utf-8'))
             pro_dob = string.replace(dob, '/', '')
             self.dobs.append(int(pro_dob))
             self.genders.append(ord(gender))

    def cluster_indices(self, cluster_number, label_array):
        return np.where(label_array == cluster_number)[0]

    def append_record(self,index):
        temp = []
        temp.append(self.lns[index])
        temp.append(self.dobs[index])
        temp.append(chr(self.genders[index]))
        temp.append(self.fns[index])
        self.final_list.append(temp)

    def get_record(self,index):
        temp = []
        temp.append(self.lns[index])
        temp.append(self.dobs[index])
        temp.append(chr(self.genders[index]))
        temp.append(self.fns[index])
        return temp

    def kmeans_clustering(self):
        self.cluster_num = len(set(self.dobs)) #num of clusters formed from date
        X=np.matrix(zip(self.dobs,self.genders))
        kmeans = KMeans(n_clusters=self.cluster_num).fit(X)
        return kmeans

    def hierarchical_clustering(self,kmeans):
        self.final_list = []
        self.final_clusters = []
        for cluster_index in set(kmeans.labels_):
            record_indices = self.cluster_indices(cluster_index, kmeans.labels_)
            names_temp = []
            for i in record_indices:
                 names_temp.append(self.names[i])
            if len(names_temp) == 1:
                 self.append_record(record_indices[0])
            elif len(names_temp) > 1:
                names_temp = np.asarray(names_temp)
                lev_similarity = -1*np.array([[jellyfish.levenshtein_distance(w1,w2) for w1 in names_temp] for w2 in names_temp])
                affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed",damping=0.5)
                affprop.fit(lev_similarity)
                #print affprop.labels_
                #affprop = KMeans(init="k-means++",n_clusters=lev_similarity.shape[0]).fit(lev_similarity)
                for affprop_cindex in set(affprop.labels_):
                    temp_indices = self.cluster_indices(affprop_cindex, affprop.labels_)
                    if len(temp_indices) > 0:
                        self.append_record(record_indices[temp_indices[0]])
                        temp_list = []
                        #print record_indices
                        #print temp_indices
                        for idx in range(len(temp_indices)):
                            temp = self.get_record(record_indices[temp_indices[idx]])
                            temp_list.append(temp)
                        self.final_clusters.append(temp_list)


    def execute(self):
        self.load_preprocess_data()
        model = self.kmeans_clustering()
        self.hierarchical_clustering(model)
        #print self.final_list
        #print len(self.final_list)
        fp = open("out.txt", "w")
        for i in range(len(self.final_list)):
            out = ""
            for j in range(len(self.final_list[i])):
                out = out + str(self.final_list[i][j]) + ","
            out = out.rstrip(',')
            fp.write(out+"\n")
        fp.close()
        fp = open("clusters.txt","w")
        fp.write("Names that formed clusters" + "\n" + "\n")
        for i in range(len(self.final_clusters)):
            fp.write("Cluster" + str(i) + "\n")
            for j in range(len(self.final_clusters[i])):
                out = ""
                for k in range(len(self.final_clusters[i][j])):
                    #print self.final_clusters[i][j][k]
                    out = out + str(self.final_clusters[i][j][k]) + ","
                out=out.rstrip(',')
                fp.write(out + "\n")
            fp.write("\n")
        fp.close()

def main():
    Deduplicate_instance = Deduplicate()
    Deduplicate_instance.execute()

if __name__== "__main__":
    main()

