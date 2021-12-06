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
Used neural net model of 42 inputs, hidden layer 1 with 80 nodes and 'relu' activation, hidden layer 2 with 30 nodes and 'relu' activation, and output layer with 1 output and 'sigmoid' activation.  Achieved ~73% accuracy in predicting success with loss of 0.55.


## Optimize the Model (Rounds 2, 3, 4)
Make some changes in the preprocessing and/or compile/train steps of the model and attempt to achieve higher percent accuracy

## Round 2 (try eliminating additional features in preprocessing):
Attempt to eliminate additional features.  Speculated that perhaps an organization's affiliation does not directly impact the success of a charity event.  For this round eliminated columns 'ORGANIZATION' and 'AFFILIATION'.  Fewer features to test after the 'get_dummies' preprocessing (32 instead of 42).  RESULT: lower accuracy at 62% with 0.63 loss.  Restore features.

## Round 3 (try beefing up the evaluation process more hidden layers, more epochs, more nodes):
Added one more hidden layer, went to 50 nodes per hidden layer and increased epochs to 200.  No change in results, accuracy achieved ~72-73% with every epoch.  Conclusion: for this case, accuracy will not improve by just beefing up the evaluation step.

## Round 4:
Add another hidden layer to the model

## Round 5:
Try using different activation functions in hidden layers