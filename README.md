## EXNO-3-DS
```
    Name: Thirumurugan. K
    Reg NO: 212224110057

```
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
FEATURE ENCODING
        
    import pandas as pd
    df=pd.read_csv("Encoding Data.csv")
    df
![image](https://github.com/user-attachments/assets/f0960a99-1db7-4050-b7c5-e7fde6e8b0bb)

    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
    pm=['Hot','Warm','Cold']
    e1=OrdinalEncoder(categories=[pm])
    e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/44609999-a59f-4c9a-823d-b632d07b4cc7)

    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
![image](https://github.com/user-attachments/assets/08301ca0-7a20-4812-8c93-fef929a40b5a)

    le=LabelEncoder()
    dfc=df.copy()
    dfc['ord_2']=le.fit_transform(dfc['ord_2'])
    dfc
![image](https://github.com/user-attachments/assets/59476503-8f64-4ea4-b363-20c3a25f6c2f)

    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder(sparse=False)
    df2=df.copy()
    enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
    df2=pd.concat([df2,enc],axis=1)
    df2
![image](https://github.com/user-attachments/assets/c9f62e63-8a40-42a1-8a12-d4e317044f1a)

    pd.get_dummies(df2,columns=["nom_0"])
![image](https://github.com/user-attachments/assets/ee550900-a77a-44de-b4f3-8117771ecc8f)

    pip install --upgrade category_encoders
![image](https://github.com/user-attachments/assets/fc661ae2-f77b-4dc0-a954-269fdb474bcb)

    from category_encoders import BinaryEncoder
    df=pd.read_csv("/content/data.csv")
    df
![image](https://github.com/user-attachments/assets/8bc56a47-da9c-4fe0-a6c6-55fc6af89337)

    be=BinaryEncoder()
    nd=be.fit_transform(df['Ord_2'])
    dfb=pd.concat([df,nd],axis=1)
    dfb1=df.copy()
    dfb
![image](https://github.com/user-attachments/assets/7ecb3430-5f16-45bc-b35e-d764d42da231)

    from category_encoders import TargetEncoder
    te=TargetEncoder()
    CC=df.copy()
    new=te.fit_transform(X=CC["City"],y=CC["Target"])
    CC=pd.concat([CC,new],axis=1)
    CC
![image](https://github.com/user-attachments/assets/c3202b53-c662-4782-8391-6b6a4b3b42a9)

FEATURE TRANSFORMATION
    
    import pandas as pd
    from scipy import stats
    import numpy as np
    df=pd.read_csv("/content/Data_to_Transform.csv")
    df
![image](https://github.com/user-attachments/assets/8b331764-4df4-4336-9b0d-eab15fc46aae)

    df.skew()
![image](https://github.com/user-attachments/assets/223b68f8-6321-4dbc-80bd-2736d936ba6d)

    np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/398ca687-4b1f-4b7b-bb8f-bd3d70275426)

    np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/3eb61c30-17bb-42e3-8cfd-0a58756efc2f)

    np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/6e3b678e-a599-42e4-8fb4-97e8e2cc49e3)

    np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/7aabb457-e3aa-4be0-b3c0-d4bc5951b2d6)

    df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
    df
![image](https://github.com/user-attachments/assets/0c2ec851-6885-4033-893c-b87756316e41)

    df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
    df
![image](https://github.com/user-attachments/assets/a6d56249-01dd-4ec5-ab39-595e7af8384a)

    df.skew()
![image](https://github.com/user-attachments/assets/42ce0204-ecd6-43ec-b2f0-8b5a33059eca)

    import seaborn as sns
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/c9021494-c526-4ed3-8ef5-8c392bdb7fc4)

    sm.qqplot(np.reciprocal(df["Moderate Negative Skew_yeojohnson"]),line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/a5ebc128-b7ab-4769-8c36-e42e83752839)

    from sklearn.preprocessing import QuantileTransformer
    qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

    df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/c487cc0c-d52e-4cca-8ffd-f5b277a98097)

    df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
    sm.qqplot(df["Highly Negative Skew"],line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/b120f486-27d6-4eba-ab23-91164f6bbad1)

    sm.qqplot(df["Highly Negative Skew_1"],line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/8fae6cb4-afa9-4f17-b73d-65b7f2a072b2)
  
# RESULT:
       Thus, the given data was successfully read, feature encoding and transformation were performed, and the resulting data was saved to a file.

