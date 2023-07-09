
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2


class preprocessing:
    def __init__(self, df):
        self.df = df

    def process(self):
        # self.df.head()
        # df.info()  # display format
        self.df = self.df.astype(str)  # convert to str
        self.df.isna().sum()
        self.df = self.df.dropna(axis=1)
        # self.df.describe()
        self.df.drop_duplicates(inplace=True)
        self.df['diagnosis'].value_counts()
        # sns.countplot(df['diagnosis'],label="count")
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 0, 'B': 1})
        ########################################################################################

    def split(self):
        x = self.df.drop('diagnosis', axis=1.0)  # remove column (diagnosis) from data
        y = self.df['diagnosis']  # include column (diagnosis)
        FeatureSelection = SelectPercentile(score_func=chi2, percentile=20)  # score_func can = f_classif
        X = FeatureSelection.fit_transform(x, y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
