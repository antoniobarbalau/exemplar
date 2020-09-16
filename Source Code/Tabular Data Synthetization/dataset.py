import pandas as pd
import numpy as np

class Adult(object):
    def __init__(
        self,
        path ='/home/tonio/research/datasets/adult.csv'
    ):
        self.dataset = pd.read_csv(path)
        self.numerical_columns = [
            'age', 'fnlwgt', 'educational-num', 'hours-per-week'
        ]
        self.linear_columns = [
            'capital-gain', 'capital-loss',
        ]
        self.categorical_columns = [
            'workclass', 'education', 'marital-status',
            'occupation', 'relationship', 'race', 'gender',
            'native-country'
        ]
        self.values = dict()
        self.means = dict()
        self.stds = dict()

        self.extract_values()

    def extract_values(self):
        for column_name in self.categorical_columns:
            self.values[column_name] = list(np.unique(self.dataset[column_name]))
        for column_name in self.numerical_columns:
            self.means[column_name] = np.mean(self.dataset[column_name])
            self.stds[column_name] = np.std(self.dataset[column_name])

    def preprocess(self, lines):
        numerical = [
            [
                float(
                    (lines.take([i])[column_name] - self.means[column_name]) /
                    self.stds[column_name]
                )
                for column_name in self.numerical_columns
            ]
            for i in range(len(lines))
        ]
        linear = [
            [
                float(lines.take([i])[column_name])
                for column_name in self.linear_columns
            ]
            for i in range(len(lines))
        ]
        categorical = [
            [
                self.values[column_name].index(lines.take([i])[column_name].item())
                for column_name in self.categorical_columns
            ]
            for i in range(len(lines))
        ]
        label = [
            0 if '<' in lines.take([i])['income'].item() else 1
            for i in range(len(lines))
        ]
        return numerical, linear, categorical, label

    def postprocess(self, numerical, linear, categorical):
        return [
            {
                'age': np.round(
                    numerical[i][0] * self.stds['age'] + self.means['age']
                ),
                'workclass': self.values['workclass'][
                    np.argmax(categorical[0][i])
                ],
                'fnlwgt': np.round(
                    numerical[i][1] * self.stds['fnlwgt'] + self.means['fnlwgt']
                ),
                'education': self.values['education'][
                    np.argmax(categorical[1][i])
                ],
                'educational-num': np.round(
                    numerical[i][2] * self.stds['educational-num'] +
                    self.means['educational-num']
                ),
                'marital-status': self.values['marital-status'][
                    np.argmax(categorical[2][i])
                ],
                'occupation': self.values['occupation'][
                    np.argmax(categorical[3][i])
                ],
                'relationship': self.values['relationship'][
                    np.argmax(categorical[4][i])
                ],
                'race': self.values['race'][
                    np.argmax(categorical[5][i])
                ],
                'gender': self.values['gender'][
                    np.argmax(categorical[6][i])
                ],
                'capital-gain': float(linear[i][0]),
                'capital-loss': float(linear[i][1]),
                'hours-per-week': np.round(
                    numerical[i][3] * self.stds['hours-per-week'] +
                    self.means['hours-per-week']
                ),
                'native-country': self.values['native-country'][
                    np.argmax(categorical[7][i])
                ]
            }
            for i in range(len(numerical))
        ]

    def to_svm(self, lines):
        numerical = [
            [
                float(
                    (lines.take([i])[column_name] - self.means[column_name]) /
                    self.stds[column_name]
                )
                for column_name in self.numerical_columns
            ]
            for i in range(len(lines))
        ]
        linear = [
            [
                float(lines.take([i])[column_name])
                for column_name in self.linear_columns
            ]
            for i in range(len(lines))
        ]
        categorical = [
            [
                list(np.eye(len(self.values[column_name]))[
                    self.values[column_name].index(lines.take([i])[column_name].item())
                ])
                for column_name in self.categorical_columns
            ]
            for i in range(len(lines))
        ]
        new_categorical = []
        for i in range(len(lines)):
            new_categorical.append([])
            for j in range(len(self.categorical_columns)):
                new_categorical[i] += categorical[i][j]
        categorical = new_categorical
        labels = [
            0 if '<' in lines.take([i])['income'].item() else 1
            for i in range(len(lines))
        ]
        return np.concatenate([
            numerical, linear, categorical
        ], axis = -1), labels

    def json_to_svm(self, lines):
        numerical = [
            [
                float(
                    (lines[i][column_name] - self.means[column_name]) /
                    self.stds[column_name]
                )
                for column_name in self.numerical_columns
            ]
            for i in range(len(lines))
        ]
        linear = [
            [
                float(lines[i][column_name])
                for column_name in self.linear_columns
            ]
            for i in range(len(lines))
        ]
        categorical = [
            [
                list(np.eye(len(self.values[column_name]))[
                    self.values[column_name].index(lines[i][column_name])
                ])
                for column_name in self.categorical_columns
            ]
            for i in range(len(lines))
        ]
        new_categorical = []
        for i in range(len(lines)):
            new_categorical.append([])
            for j in range(len(self.categorical_columns)):
                new_categorical[i] += categorical[i][j]
        categorical = new_categorical
        return np.concatenate([
            numerical, linear, categorical
        ], axis = -1)

