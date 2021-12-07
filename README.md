## BINARY CLASSIFIER MODEL for CHARITY FUNDING PREDICTIONS

# charity-funding-predictor Overview
The purpose of this analysis is to build a binary classifier (Machine Learning) for Alphabet Soup organization that can determine whether an applicant will be successful in its fundraising given numerous features from the application.  Each applicant asked for funding (ask_amt) and indicates its income amount.  Without knowing the amount raised, we have a feature called IS_SUCCESSFUL that we can use as our label data to train and test our model to see how well we can predict whether a new applicant will be successful given only the application data.


## Data preprocessing
Used conventional methods to preprocess the data: 
1. eliminating features that do not contribute to the fundraiser successes;

2. setting up bins for other features with default bin 'Other' to group the lower frequency numeric values;

3. use label encoder to reduce the 'Y' and 'N' values to 1 and 0 in SPECIAL_CONSIDERATIONS feature;

4. convert remaining categorical data to numerical data using pd.get_dummies()

5. subdivide the dataset into training and testing data, use StandardScaler on the training set only

## Compile, train, evaluate the Model (Round 1)
Used neural net model of 42 inputs, hidden layer 1 with 80 nodes and 'relu' activation, hidden layer 2 with 30 nodes and 'relu' activation, and output layer with 1 output and 'sigmoid' activation.  Achieved ~73% accuracy in predicting success with loss of 0.55.


## Optimize the Model (Rounds 2, 3, 4, 5) (AlphabetSoupCharity_Optimization)
Attempt different tweaks to attempt to improve accuracy above 72%

## Round 2 (try eliminating additional features in preprocessing):
Attempt to eliminate additional features.  Speculated that perhaps an organization's affiliation does not directly impact the success of a charity event.  For this round eliminated columns 'ORGANIZATION' and 'AFFILIATION'.  Fewer features to test after the 'get_dummies' preprocessing (32 instead of 42).  RESULT: lower accuracy at 62% with 0.63 loss.  Conclusion: Restore features.

## Round 3 (try beefing up the evaluation process more hidden layers, more epochs, more nodes):
Added one more hidden layer, went to 50 nodes per hidden layer and increased epochs to 200.  No change in results, accuracy achieved ~72-73% with every epoch.  Conclusion: for this case, accuracy will not improve by just beefing up the evaluation step.

## Round 4 (try a different activation algorithm 'tanh')
Attempted to try different activation algorithm in evaluation step.  Used 'tanh' combination with 'relu' as well as exclusively 'tanh'.  Result: achieved ~73% accuracy in predicting succes with loss of 0.56.  This did not have an impact on improving the model.

## REGARDING EXTRA FEATURES
These features must remain: application_type, classification, affiliation.  When removing them, predictions fall into the 60s. So they must be included.

For these next features, when I removed them in preprocessing, we would continue to receive 73% accuracy; that likely means these features are not required for making the same 73% accuracy.  It does reduce the number of features.
'SPECIAL_CONSIDERATIONS', 'USE_CASE', 'INCOME_AMT', 'STATUS', 'ASK_AMT', 'ORGANIZATION'

## CONCLUSION and Round 5 (combine all previous enhancement attempts):
With as few features as possible and as much binning as possible, with additional hidden layer, doubling epoch count and doubling the number of nodes per hidden layer, we continue to achieve only 72-75% accuracy.  
* 'SPECIAL_CONSIDERATIONS', 'USE_CASE', 'INCOME_AMT', 'STATUS', 'ASK_AMT', 'ORGANIZATION' can be removed
* APPLICATION_TYPE, CLASSIFICATION, and AFFILIATION are the key features 
* IS_SUCCESSFUL is the target
* For compiling/training/evaluating: tried doubling the neuron counts, halving the neuron counts, adding 1-2 additional hidden layers all for the purposes of beefing up the evaluation engine.  Attempted 'tanh' and 'relu' activation functions in case one dominated over the other.

## SUMMARY
The overall result indicates that there were many features in the dataset that did not contribute to the predictions of success.  As witnessed by dropping of the above mentioned features, we were able to perform dimensionality reduction as well with binning, standard scaling.  We might just be missing some outlier data that could be consistently confusing the model.  One possible method we could use in the future might be PCA (Principal Component Analysis) to confirm the more important features that are important to the categorization while possibly removing low impact features.