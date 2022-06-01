import numpy as np
from sklearn.base import BaseEstimator
import time


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    # YOUR CODE HERE
    m = np.sum(np.sum(y))
    s = np.sum(y,axis = 0) / m
    
    return - s @ np.log2(s+EPS)
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    
    # YOUR CODE HERE
    m = np.sum(np.sum(y))
    s = np.sum(y,axis = 0) / m
    
    return 1 - np.dot(s,s)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    
    return np.sum((y - np.mean(y))**2) / len(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return np.sum( abs(y - np.median(y)) ) / len(y)


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return np.reshape((y_one_hot.argmax(axis = 1)[:, None]),(len(y_one_hot)))


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index = -10, threshold = -10, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        self.classes = None
       
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        
        
        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
        
        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        n_objects = np.shape(X_subset)[0]
        
        
        
        #for i in range(n_objects):
        #    if X_subset[i,feature_index]<threshold:
        #        X_left.append(X_subset[i])
        #        y_left.append(y_subset[i])
        #    else:
        #        X_right.append(X_subset[i])
        #        y_right.append(y_subset[i])
 
        #X_left,X_right = [X_subset[X_subset[:,feature_index]<threshold], X_subset[X_subset[:,feature_index]>=threshold]]
        #y_left,y_right = [y_subset[X_subset[:,feature_index]<threshold], y_subset[X_subset[:,feature_index]>=threshold]]
        
        X_left = np.array([X_subset[i] for i in range(n_objects) if X_subset[i,feature_index]<threshold])
        y_left = np.array([y_subset[i] for i in range(n_objects) if X_subset[i,feature_index]<threshold])
        
        X_right = np.array([X_subset[i] for i in range(n_objects) if X_subset[i,feature_index]>=threshold])
        y_right = np.array([y_subset[i] for i in range(n_objects) if X_subset[i,feature_index]>=threshold])
        

        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        n_objects = np.shape(X_subset)[0]
        
        y_left = np.array([y_subset[i] for i in range(n_objects) if X_subset[i,feature_index]<threshold])
        y_right = np.array([y_subset[i] for i in range(n_objects) if X_subset[i,feature_index]>=threshold])
        
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE 
        n_objects, n_features = np.shape(X_subset)[0], np.shape(X_subset)[1]
        
        method = self.all_criterions[self.criterion_name][0]
        H = method(y_subset)
        
        G_max = 0
        
        if np.max(np.sum(y_subset,axis=0))/np.shape(y_subset)[0] != 1.0:
            for index in range(n_features):
                holder = np.unique(X_subset[:,index])
                        
                for t in holder:
                    if t != max(holder) and t != min(holder):          
                        y_l, y_r = self.make_split_only_y(index, t, X_subset, y_subset)
                        if np.shape(y_l)[0] >= self.min_samples_split and np.shape(y_r)[0] >= self.min_samples_split:
                            G = H - method(y_l) * (np.shape(y_l)[0]/n_objects) - method(y_r) * (np.shape(y_r)[0]/n_objects)
     
                            if G > G_max:
                                G_max = G
                                feature_index, threshold = index, t
            if G_max == 0:
                return -1,-1
        else:
            return -1,-1
          
        
               
               
        return feature_index, threshold
 
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if self.depth < self.max_depth and self.min_samples_split *2 < np.shape(y_subset)[0]:
            feature_index, threshold = self.choose_best_split(X_subset, y_subset)
            
            if feature_index == -1 and threshold == -1:
                a = Node()
                a.classes = np.argmax(np.bincount(one_hot_decode(y_subset)))
                a.proba = np.sum(y_subset, axis=0) / y_subset.shape[0]
                if not self.all_criterions[self.criterion_name][1]:
                    a.classes = np.mean(y_subset)
                return a
            
            new_node = Node(feature_index, threshold,proba=0)
            
            (X_l, y_l), (X_r, y_r) = self.make_split(new_node.feature_index, new_node.value, X_subset, y_subset)
            self.depth+=1
            new_node.left_child = self.make_tree(X_l, y_l)
            new_node.right_child = self.make_tree(X_r, y_r)
            self.depth-=1
            return new_node
        else:
            a = Node()
            if self.all_criterions[self.criterion_name][1]:
                a.classes = np.argmax(np.bincount(one_hot_decode(y_subset)))
                a.proba = np.sum(y_subset, axis=0) / y_subset.shape[0]
            else:
                a.classes = np.mean(y_subset)
            return a
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
        
        
        self.root = self.make_tree(X, y)
        
    
    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        n_objects = np.shape(X)[0]
        y_predicted = np.zeros((n_objects,1),dtype=int)
        
        for i in range(n_objects):
            cur_node = self.root
            while(cur_node.left_child != None):
                if X[i][cur_node.feature_index] < cur_node.value:      
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            y_predicted[i][0] = cur_node.classes
        
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        n_objects = np.shape(X)[0]
        y_predicted_probs = np.zeros((n_objects,self.n_classes),dtype=int)
        
        for i in range(n_objects):
            cur_node = self.root
            while(cur_node.left_child != None):
                if X[i][cur_node.feature_index] < cur_node.value:      
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            y_predicted_probs[i] = cur_node.proba
                    
        
        
        return y_predicted_probs
