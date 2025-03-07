data = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()

data.columns

data.info()

data.describe()

from ydata_profiling import ProfileReport
profile = ProfileReport(data, title="Churn Data")
