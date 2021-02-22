import numpy as np
import pandas as pd
from models.utilities.normalization import normal_norm


class KNN:

    def __init__(self, data, algo_name="knn_weighted_psm", predict_label='label', forecast_input_attributes=0):

        self.data_df = pd.read_excel(data)
        self.data_df = self.data_df.dropna()

        self.predict = predict_label
        self.algorithm = algo_name  # algorithm used to change getNeighbour in kNN predict function
        self.data_df = data
        self.output = np.array([])

        if self.algorithm == "knn_weighted_psm":
            print("applying kNN_ weighted_psm")

        if self.algorithm == "knn_psm":
            print("applying kNN_Unweighted_psm")

        if self.algorithm == "knn_np":
            print("applying kNN_np")

        self.x = np.array(self.data_df.iloc[:, :-1])  # X is the feature data without last column of the dataset
        self.y = np.array(self.data_df.iloc[:, -1])

        # self.x = np.array(self.data_df.drop([self.predict], 1))
        self.x = np.asarray(self.x).astype(np.float64)
        # self.y = np.array(self.data_df[self.predict])

        self.n_sample, n_features = np.shape(self.x)
        self.n_labels = np.size(np.unique(self.y))
        self.n_train = self.n_sample

        self.x_norm = np.copy(self.x)
        # self.x_norm[:, 2:] = self.normal_norm(np.asarray(self.x_norm[:, 2:]).astype(np.float64))
        self.x_norm[:, 2:] = normal_norm(np.asarray(self.x_norm[:, 2:]).astype(np.float64))
        # Running on Training Data

        self.k = int(pow(self.n_train, 0.5)) if int(pow(self.n_train, 0.5)) % 2 != 0 else int(
            pow(self.n_train, 0.5)) + 1

        for i in range(0, self.n_train):
            self.pred = self.kNN_predict(self.x_norm[i], self.k, self.x_norm, self.n_train, self.n_labels)
            self.output = np.append(self.output, self.pred)
        self.acc = np.sum(self.output == self.y[:self.n_train]) / self.n_train
        train_result = 100 * self.acc
        self.train_acc = 100 * round(self.acc, 3)
        print('Train Data Accuracy : ' + str(self.train_acc))

        # Running on Test Data
        self.n_test = len(self.test)
        self.output = np.array([])

        for i in range(self.n_train, self.n_sample):
            self.pred = self.kNN_predict(self.x_norm[i], self.k, self.x_norm, self.n_train, self.n_labels)

            self.output = np.append(self.output, self.pred)

        self.acc = np.sum(self.output == self.y[self.n_train:]) / self.n_test
        test_result = 100 * self.acc

        self.test_acc = 100 * round(self.acc, 3)
        print('Test Data Accuracy : ' + str(self.test_acc))

        if forecast_input_attributes != 0:
            forecast_result = self.forecast(forecast_input_attributes, self.x)
            print(forecast_result)

    def euclideanDistance(self, a, b):
        """
        Calculates the Euclidiean distance between two points
        :param a: first point
        :param b: second point
        :return: euclidean distance between first point and the second point
        """
        dist = np.linalg.norm(a - b)
        return dist

    def Srt_np(self, distances):
        distances.sort()
        return distances

    # functions  to get distances with all the data points from the test_out data point
    def getNeighbours(self, testdata, x, n_sample):
        similarity = []
        for i in range(0, n_sample):
            similarity.append([self.euclideanDistance(x[i], testdata), self.y[i]])
        return self.Srt_np(similarity)

    # compute patient similarity matrix of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    # this is the direction/orientation similarity
    def PSM(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v1))

    # Shorting the similarity in descending order
    def Srt(self, similarity):
        # return np.flip(np.sort(similarity, axis=None, kind='mergesort'))
        similarity.sort(reverse=True)
        return similarity

    # Feature similarity for sex
    def FS_S(self, a, b):
        if a == b:
            return 1
        else:
            return 0

    # Feature similarity 
    def FS_A(self, a, b):
        return (min(a, b) / max(a, b))

    def getNeighbours_weighted(self, testdata, x, n_sample):
        W = [0.4, 0.1, 0.1]  # weight as per the Wang et al. BioMed Eng OnLine paper
        similarity = []
        for i in range(0, n_sample):
            temp = W[1] * self.FS_S(x[i, 0], testdata[0]) + W[2] * self.FS_A(x[i, 1], testdata[1]) + W[0] * self.PSM(
                x[i, 2:], testdata[2:])
            similarity.append([temp, self.y[i]])
        return self.Srt(similarity)

    def getNeighbours_unWeighted(self, testdata, x, n_sample):
        similarity = []
        for i in range(0, n_sample):
            similarity.append([self.PSM(x[i], testdata), self.y[i]])
        return self.Srt(similarity)

    def kNN_predict(self, testdata, k, x, n_train, n_labels):

        if self.algorithm == "knn_weighted_psm":
            # print("code block of kNN_ weighted_psm")
            similarity = self.getNeighbours_weighted(testdata, x, n_train)

        if self.algorithm == "knn_psm":
            # print("code block of kNN_ weighted_psm")
            similarity = self.getNeighbours_unWeighted(testdata, x, n_train)

        if self.algorithm == "knn_np":
            similarity = self.getNeighbours(testdata, x, n_train)

        # similarity = self.getNeighbours(testdata,x,n_train)
        out = list(np.zeros(n_labels))
        for i in range(0, k):
            out[similarity[i][1]] += 1
        mx_i = 0
        for i in range(0, n_labels):
            if out[mx_i] < out[i]:
                mx_i = i
        return mx_i

    def forecast(self, x, arr):
        temp = np.asarray(arr[:, 2:]).astype(np.float64)
        x[2:] = (x[2:] - temp.mean(axis=0)) / temp.std(axis=0)
        return self.kNN_predict(x, self.k, self.x_norm, self.n_train, self.n_labels)
