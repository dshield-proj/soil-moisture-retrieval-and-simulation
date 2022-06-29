from datetime import datetime
from json import JSONEncoder
import numpy as np
import pandas as pd


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime) or isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (complex, np.complex_)):
            return str(obj).strip('(').strip(')')
        return JSONEncoder.default(self, obj)
