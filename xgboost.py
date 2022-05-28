import json
import sys
import pandas as pd
import xgboost as xgb
from pyspark import SparkContext

def conf_spark():
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    return sc

def train_model(trainRDD, businessRDD, userRDD):
    finalRDD = trainRDD.join(businessRDD).map(
        lambda line: (line[1][0][0], (line[1][1][2], line[1][1][3], line[1][0][1]))).join(userRDD)\
        .map(lambda line: (line[1][0][0], line[1][0][1], line[1][1][0], line[1][1][1], float(line[1][0][2]))).collect()

    # 'State', 'City', 'Business Stars', 'Business Review Count', 'User Review Count', 'User Average Stars', 'Stars'
    df = pd.DataFrame(finalRDD,
                      columns=['Business Stars', 'Business Review Count', 'User Review Count', 'User Average Stars',
                               'Stars'])
    X_train, y_train = df.iloc[:, :-1], df.iloc[:, -1]
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model

def predict(model, testRDD, businessRDD, userRDD):
    finalRDD = testRDD.join(businessRDD).map(
        lambda line: (line[1][0], (line[0], line[1][1][0], line[1][1][1], line[1][1][2], line[1][1][3]))).join(userRDD)\
        .map(
        lambda line: (line[0], line[1][0][0], line[1][0][3], line[1][0][4], line[1][1][0], line[1][1][1])).collect()
    # print(testRDD)
    df_test = pd.DataFrame(finalRDD, columns=['User Id', 'Business Id', 'Business Stars', 'Business Review Count', 'User Review Count', 'User Average Stars'])

    X_test = df_test[['Business Stars', 'Business Review Count', 'User Review Count',
                      'User Average Stars']]
    user_business_cols = df_test[['User Id', 'Business Id']]
    preds = model.predict(X_test)
    return preds, user_business_cols


if __name__ == '__main__':
    sc = conf_spark()
    train = sc.textFile(sys.argv[1] + "/" + "yelp_train.csv")
    headers = train.first()
    train = train.filter(lambda line: line != headers)
    train = train.map(lambda line: (line.split(",")[1], (line.split(",")[0], line.split(",")[2])))

    #business details
    business = sc.textFile(sys.argv[1] + "/" + "business.json").map(json.loads).map(lambda line: (line['business_id'], (line['city'], line['state'], line['stars'], line['review_count'])))

    #user details
    user = sc.textFile(sys.argv[1] + "/" + "user.json").map(json.loads).map(lambda line: (line['user_id'], (line['review_count'], line['average_stars'])))

    test = sc.textFile(sys.argv[2])
    t_headers = test.first()
    test = test.filter(lambda line: line != t_headers).map(lambda line: (line.split(",")[1], (line.split(",")[0])))

    model = train_model(train, business, user)
    preds, user_business_cols = predict(model, test, business, user)

    with open(sys.argv[3], 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for _ in range(len(preds)):
            f.write(user_business_cols['User Id'][_] + "," + user_business_cols['Business Id'][_] + "," + str(preds[_]) + "\n")

