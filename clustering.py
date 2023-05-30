from matplotlib import pyplot as plt
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import math
from sklearn.metrics import silhouette_samples, silhouette_score

def clustering_filemaker(df, labels, task_n):
    cluster_numbers = [10,50,100]
    events = np.unique(labels)
    predDICT = {10:{},50:{},100:{}}

    labels = labels.to_numpy()
    #df = df.drop('label', axis = 1)
    balanced = df.copy()

    for event in events:
        clusters = {}

        for cluster_number in cluster_numbers:
            for i in range(0, cluster_number):
                clusters[i] = {0:0, 1:0, 2:0, 3:0}
            perclist = []
            kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init="auto").fit(balanced)
            predicted = kmeans.labels_
            centroids  = kmeans.cluster_centers_
            
            for i, cluster in enumerate(predicted):
                clusters[cluster][labels[i]] += 1

            for i in clusters:
                if (max(clusters[i].values()) > 50):
                    fig = plt.bar([0,1,2,3], clusters[i].values(), color='b')
                    plt.title("Number of label occurences in cluster %i, task %s, number of clusters %i"%(i,task_n,cluster_number))
                    plt.xticks(range(0,4))
                    plt.xlabel("Label")
                    plt.ylabel("Number of occurences in cluster")
                    plt.savefig('./plots/clustering/t_%s/c_%i/cluster_%i.png'%((task_n,cluster_number,i)))
                    plt.clf()
                
                    perc = round(100* (max(clusters[i].values())/ sum(clusters[i].values())))
                    if (max(clusters[i].values()) > 50):
                        perclist.append([i, clusters[i], max(clusters[i], key=clusters[i].get), str(perc)+ '%', centroids[i]])

            with open ('./plots/clustering/t_%s/c_%i/percentages.csv'%((task_n,cluster_number)), 'w') as f:
                write = csv.writer(f)
                write.writerows([['cluster', 'label distribution', 'label', 'label percentage', 'cluster center']])
                write.writerows(perclist)

            dist_dict = {}
            for i, cluster in enumerate(clusters.values()):
                dist_dict[i] = (cluster[event])

            distlist = list(dist_dict.values())
            #stacked_line[event] = distlist
            predDICT[cluster_number][event] = distlist

            plt.bar(range(0,cluster_number), distlist)
            plt.ylabel("Number of datapoints")
            plt.xlabel("Cluster")
            plt.title('number of labels per cluster, task %s, event %i'%(task_n, event))
            plt.savefig('./plots/clustering/t_%s/c_%i/event_%i'%(task_n, cluster_number, event))
            plt.clf()

def predict(dataset, perclist):
    predicted = []

    for datapoint in dataset.values:
        best = 999
        for i in perclist:
            if 1-i[3] * math.dist(datapoint, i[4]) < best:
                best = 1-i[3] * math.dist(datapoint, i[4])
                best_cluster = i[2]  
        predicted.append(best_cluster)
    return predicted

def convert_perc_to_ratio(clusters):
    cluster_perc = []
    for i in clusters['label percentage'].values:
        cluster_perc.append(int(i.replace('%',''))/100)

    clusters['label percentage'] = cluster_perc
    return clusters

def silhouettescore_plot(df, task_n):
    X = df.values

    range_n_clusters = [10, 50, 100]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])



        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        plt.savefig('./plots/clustering/t_%s/silhouettes_%d'%(task_n,n_clusters))
