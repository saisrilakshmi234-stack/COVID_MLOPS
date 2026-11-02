import numpy as np
import pandas as pd


def preprocess_request(request, preprocessor):
    input_data = request.model_dump(by_alias=True)

    expected_columns = list(preprocessor.feature_names_in_)

    dataframe = pd.DataFrame([input_data])
    dataframe = dataframe[expected_columns]

    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    transformed_data = preprocessor.transform(dataframe)

    return transformed_data
