[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#unsupervised-gaze-event-discrimination">Unsupervised-Gaze-Event-Discrimination</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgments</a></li>
  </ol>
</details>

# Unsupervised-Gaze-Event-Discrimination
This repository contains the supplementary code for the research paper ['link']. In this project, we researched the possibilities of the use of unsupervised machine learning for the classification of gaze events in mobile eye-trackers. The method we proposed did not show sufficient performance, but we believe that the code provided in this repository is a versatile toolbox, which could be used for further research.

## Usage
### installation
The code could be downloaded by 

### example
In the document, you can find a file called 'Main.IPYNB'. This file contains an example of how the toolbox could be used. In this file, we explore the preprocessing, and clustering functions.

### preprocessing.py
The preprocessing module contains functions to open and preproces the data. The following functions could be found:
* **raw_file_labeler_opener** used to open the raw data files from the Gaze-In-Wild[1] dataset and returns them in a way useable for further research.
* **file_opener** used to open files generated by the ACE-DNV[2] method.
* **normalize** custom made normalization function, which normalizes over each column individually, instead of normalizing over the entire dataset, like other normalization functions do.
* **random_undersampler** dunction for randomly undersampling the data
* **DC_data_balancer** undersampling mbalancer using divide and conquer technique, for increased speed.[2]
* **get_frames** Function that randomly undersamples an x number of datapoints for each gaze event from each task. this method ensures the highest variance of tasks, while undersampling.
* **majority_vote** Function that assigns a label to a datapoint, when the majority of labelers voted for this label. with various options to increase cofidence, decrease the number of discarded datapoints and behaviour when number of votes are equal.
* **agreement** Function that extracts all datapoints, where all labelers agree upon a label.

### clustering.py
The clustering module contains functions, used to perform the clustering, prediction making and plotting the sillhouette score plots. The following functions could be found:
* **clustering_filemaker** This function clusters data and generates a CSV-file, containing The clusternumber, label distribution, label, percentage of majority label and the coordinates of the cluster center. The generated file is used for further analysis of the clustering performance.
* **predict** This function predicts to which cluster a new datapoint belongs, using the least eucledian distance and the confidence in that clusters.
* **convert_perc_to_ratio** Function to covert the percentage strings to ratios of the clusters CSV-file. (Example: '74%' to 0.74)
* **silhouette score plot** Plots the silhouette score plot for the dataset, with 10, 50, and 100 clusters.

# contributing
The open source community thrives on contributions, which are the driving force behind its incredible environment for learning, inspiring, and creating. Any contributions you make are highly valued. If you have any suggestions to enhance this project, we encourage you to fork the repository and submit a pull request. Alternatively, you can simply open an issue with the "enhancement" tag. Don't forget to show your support by giving the project a star! Thank you once again!

### Steps to Contribute:
1. Fork the project
2. Create a new branch for your feature (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push the changes to your branch (git push origin feature/AmazingFeature)
5. Open a pull request

## license

## contact

## acknowledgements
