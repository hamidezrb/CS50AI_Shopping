import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
months = {'jan': 0,'feb': 1,'mar': 2,'apr':3,'may':4,'june':5,'jul':6,'aug':7,'sep':8,'oct':9,'nov':10,'dec':11}

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")




def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    dict_shopping = []
    with open(filename) as f:
        # reader = csv.reader(f)
        # next(reader)
        reader = csv.DictReader(f)
        for row in reader:
            
            evidence = []
            evidence.append(int(row["Administrative"]))
            evidence.append(float(row["Administrative_Duration"]))
            evidence.append(int(row["Informational"]))
            evidence.append(float(row["Informational_Duration"]))
            evidence.append(int(row["ProductRelated"]))
            evidence.append(float(row["ProductRelated_Duration"]))
            evidence.append(float(row["BounceRates"]))
            evidence.append(float(row["ExitRates"]))
            evidence.append(float(row["PageValues"]))
            evidence.append(float(row["SpecialDay"]))
            evidence.append(int(months[row["Month"].lower()]))
            evidence.append(int(row["OperatingSystems"]))
            evidence.append(int(row["Browser"]))
            evidence.append(int(row["Region"]))
            evidence.append(int(row["TrafficType"]))
            evidence.append(int(row["VisitorType"].lower() == 'returning_visitor'))
            evidence.append(int(row["Weekend"].lower() == 'true'))
            dict_shopping.append({
            "evidence": evidence,
            "label": 1  if row["Revenue"].lower() == 'true' else 0
            })
            
    evidence = [row["evidence"] for row in dict_shopping]
    labels = [row["label"] for row in dict_shopping]
    return (evidence , labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create the k-NN classifier with k=1
    knn = KNeighborsClassifier(1)
    # Train the classifier using the training set
    knn.fit(evidence, labels)

    return knn

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_labels = [label for label in labels if label == 1]
    negative_labels = [label for label in labels if label == 0]
    sum_positive = 0
    sum_negative = 0
    for i in range(len(labels)):
       if predictions[i] == labels[i]:
           if labels[i] == 1:
               sum_positive += 1
           elif labels[i] == 0:
               sum_negative += 1  
                     
    sensitivity = sum_positive / len(positive_labels)
    specificity = sum_negative / len(negative_labels)
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
