from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from xgboost import plot_importance

def train_behavioral_model(df):
    """
    Train an XGBoost model to predict whether market reaction is positive,
    negative, or neutral based on engineered behavioral features.
    """

    # Define which columns are your input features (X)
    X = df[['price_change','volatility','loss_aversion_score','reaction_speed','herding_index']]

    # Define your target variable (what you're predicting)
    y = df['reaction_label']

    # Split data into training (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize an XGBoost classifier
    model = XGBClassifier(
        objective='multi:softmax', # for multi-class classification
        num_class=3,               # three classes: positive, negative, neutral
        eval_metric='mlogloss',   # evaluation metric
        use_label_encoder=False,    # to avoid warnings
        random_state=42
    )

    # Train the model using training data
    model.fit(X_train, y_train)

    # Test the model by predicting on the test data
    y_pred = model.predict(X_test)

    # Print accuracy metrics (how well the model did)
    print(classification_report(y_test, y_pred))

    # Visualize feature importance
    plot_importance(model)
    plt.tight_layout()
    plt.show()

    # Return the trained model (for predictions later)
    return model


