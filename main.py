import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import seaborn as sn
import matplotlib.pyplot as plt
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import seaborn as sn
import matplotlib.pyplot as plt

class DiseasePrediction:
    def _init_(self, model_name=None):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        self.verbose = self.config.get('verbose', False)
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_dataset(self.config['dataset']['training_data_path'])
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_dataset(self.config['dataset']['test_data_path'])
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # Model Definition
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config.get('model_save_path', './models/')

    def _load_dataset(self, path):
        df = pd.read_csv(path)
        cols = df.columns[:-2]
        features = df[cols]
        labels = df['prognosis']

        assert len(features.iloc[0]) == 132
        assert len(labels) == features.shape[0]

        if self.verbose:
            print("Length of Data: ", df.shape)
            print("Features: ", features.shape)
            print("Labels: ", labels.shape)
        return features, labels, df

    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config.get('random_state', 42))

        if self.verbose:
            print("Number of Training Samples: {0}\tNumber of Validation Samples: {1}".format(len(X_train), len(X_val)))
        return X_train, y_train, X_val, y_val

    def select_model(self):
        if self.model_name == 'mnb':
            return MultinomialNB()
        elif self.model_name == 'decision_tree':
            return DecisionTreeClassifier(criterion=self.config['model']['decision_tree'].get('criterion', 'gini'))
        elif self.model_name == 'random_forest':
            return RandomForestClassifier(n_estimators=self.config['model']['random_forest'].get('n_estimators', 100))
        elif self.model_name == 'gradient_boost':
            return GradientBoostingClassifier(
                n_estimators=self.config['model']['gradient_boost'].get('n_estimators', 100),
                criterion=self.config['model']['gradient_boost'].get('criterion', 'friedman_mse')
            )

    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Class Weighting
        class_weights = dict(zip(range(len(y_train.unique())), compute_class_weight('balanced', classes=y_train.unique(), y=y_train)))
        classifier = self.select_model()

        # Oversampling
        smote = SMOTE(sampling_strategy='auto')
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Training the Model
        classifier.fit(X_train_resampled, y_train_resampled)

        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val_scaled, y_val)

        # Validation Data Prediction
        y_pred = classifier.predict(X_val_scaled)

        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)

        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)

        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)

        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val_scaled, y_val, cv=5)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")

        if test_data is not None:
            result = clf.predict(test_data)
        else:
            # Feature Scaling for Test Data
            test_data_scaled = scaler.transform(self.test_features)

            # Test Data Prediction
            result = clf.predict(test_data_scaled)

            # Evaluation on Test Data
            accuracy = accuracy_score(self.test_labels, result)
            clf_report = classification_report(self.test_labels, result)
            return accuracy, clf_report

        return result


if __name__ == "_main_":
    # Model Currently Training
    current_model_name = 'gradient_boost'
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)