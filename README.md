# charity-funding-predictor
Build a binary classifier (Machine Learning) for Alphabet Soup that determines whether an applicant will be successful.  Attempt to redesign the model in order to achieve a higher prediction success percentage.

## Preprocess the data (Round 1)
Used conventional methods to preprocess the data: 
1. eliminating features that do not contribute to the fundraiser successes;

2. setting up bins for other features with default bin 'Other' to group the lower frequency numeric values;

3. use label encoder to reduce the 'Y' and 'N' values to 1 and 0 in SPECIAL_CONSIDERATIONS feature;

4. convert remaining categorical data to numerical data using pd.get_dummies()

5. subdivide the dataset into training and testing data, use StandardScaler on the training set only

## Compile, train, evaluate the Model
Used neural net model of 42 inputs, hidden layer 1 with 80 nodes and 'relu' activation, hidden layer 2 with 30 nodes and 'relu' activation, and output layer with 1 output and 'sigmoid' activation.  Achieved ~73% accuracy in predicting success.


## Optimize the Model (Rounds 2, 3, 4)
Make some changes in the preprocessing and/or compile/train steps of the model and attempt to achieve higher percent accuracy

## Round 2:
Attempt to find additional features that can be eliminated

## Round 3:
Add more nodes to hidden layer 2, adjust the number of epochs

## Round 4:
Add another hidden layer to the model

## Round 5:
Try using different activation functions in hidden layers